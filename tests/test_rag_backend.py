import json
import os
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_backend import RAGPipeline, _normalize_azure_endpoint


# ---------------------------------------------------------------------------
# _normalize_azure_endpoint (standalone function, no mocking needed)
# ---------------------------------------------------------------------------
class TestNormalizeAzureEndpoint:
    def test_strips_trailing_slash(self):
        assert _normalize_azure_endpoint("https://foo.openai.azure.com/") == "https://foo.openai.azure.com"

    def test_removes_openai_suffix(self):
        assert _normalize_azure_endpoint("https://foo.openai.azure.com/openai") == "https://foo.openai.azure.com"

    def test_removes_openai_suffix_case_insensitive(self):
        assert _normalize_azure_endpoint("https://foo.openai.azure.com/OpenAI") == "https://foo.openai.azure.com"

    def test_clean_endpoint_unchanged(self):
        assert _normalize_azure_endpoint("https://foo.openai.azure.com") == "https://foo.openai.azure.com"

    def test_empty_string(self):
        assert _normalize_azure_endpoint("") == ""

    def test_none(self):
        assert _normalize_azure_endpoint(None) == ""

    def test_whitespace(self):
        assert _normalize_azure_endpoint("  https://foo.openai.azure.com  ") == "https://foo.openai.azure.com"


# ---------------------------------------------------------------------------
# Helper to build a RAGPipeline with all heavy dependencies mocked
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_API_MODE", "chat_completions")


@pytest.fixture
def pipeline(mock_env, tmp_path):
    """Create a RAGPipeline with mocked embedding model and OpenAI client."""
    with (
        patch("rag_backend.SentenceTransformer") as MockST,
        patch("rag_backend.AzureOpenAI") as MockAOAI,
    ):
        mock_model = MagicMock()
        # encode() returns 384-dim vectors
        mock_model.encode.side_effect = lambda texts, **kw: np.random.rand(
            len(texts) if isinstance(texts, list) else 1, 384
        ).astype("float32")
        MockST.return_value = mock_model

        mock_client = MagicMock()
        MockAOAI.return_value = mock_client

        pipe = RAGPipeline()
        pipe.store_path = str(tmp_path / "vector_store")
        os.makedirs(pipe.store_path, exist_ok=True)
        yield pipe


# ---------------------------------------------------------------------------
# RAGPipeline.__init__ validation
# ---------------------------------------------------------------------------
class TestRAGPipelineInit:
    def test_raises_without_endpoint(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        with patch("rag_backend.SentenceTransformer"):
            with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
                RAGPipeline()

    def test_raises_without_deployment_name(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_NAME", raising=False)
        with patch("rag_backend.SentenceTransformer"):
            with pytest.raises(ValueError, match="AZURE_OPENAI_DEPLOYMENT_NAME"):
                RAGPipeline()

    def test_raises_on_invalid_api_mode(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_API_MODE", "invalid_mode")
        with patch("rag_backend.SentenceTransformer"):
            with pytest.raises(ValueError, match="AZURE_OPENAI_API_MODE"):
                RAGPipeline()

    def test_api_mode_alias_chat(self, mock_env, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_MODE", "chat")
        with (
            patch("rag_backend.SentenceTransformer"),
            patch("rag_backend.AzureOpenAI"),
        ):
            pipe = RAGPipeline()
            assert pipe.api_mode == "chat_completions"

    def test_api_mode_alias_responses(self, mock_env, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_MODE", "response")
        with (
            patch("rag_backend.SentenceTransformer"),
            patch("rag_backend.AzureOpenAI"),
        ):
            pipe = RAGPipeline()
            assert pipe.api_mode == "responses"


# ---------------------------------------------------------------------------
# _tokenize_chunk
# ---------------------------------------------------------------------------
class TestTokenizeChunk:
    def test_lowercases_and_splits(self, pipeline):
        tokens = pipeline._tokenize_chunk("Hello World")
        assert tokens == ["hello", "world"]

    def test_removes_punctuation(self, pipeline):
        tokens = pipeline._tokenize_chunk("Hello, world! How's it?")
        assert "hello" in tokens
        assert "world" in tokens
        # Punctuation removed, but contractions lose apostrophe
        assert all("," not in t and "!" not in t for t in tokens)

    def test_empty_string(self, pipeline):
        assert pipeline._tokenize_chunk("") == []


# ---------------------------------------------------------------------------
# _rrf_fusion
# ---------------------------------------------------------------------------
class TestRRFFusion:
    def test_merges_results(self, pipeline):
        vector = {0: 0, 1: 1, 2: 2}  # index: rank
        keyword = {1: 0, 3: 1, 4: 2}
        merged = pipeline._rrf_fusion(vector, keyword, k=3)

        assert len(merged) == 3
        # Index 1 appears in both, should rank highest
        assert merged[0] == 1

    def test_returns_k_results(self, pipeline):
        vector = {i: i for i in range(10)}
        keyword = {i + 5: i for i in range(10)}
        merged = pipeline._rrf_fusion(vector, keyword, k=5)

        assert len(merged) == 5

    def test_disjoint_results(self, pipeline):
        vector = {0: 0, 1: 1}
        keyword = {2: 0, 3: 1}
        merged = pipeline._rrf_fusion(vector, keyword, k=4)

        assert set(merged) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# _extract_text_from_response
# ---------------------------------------------------------------------------
class TestExtractTextFromResponse:
    def test_chat_completions_response(self):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "The answer is 42."
        # Remove output_text so it doesn't trigger the responses branch
        del response.output_text

        text = RAGPipeline._extract_text_from_response(response)
        assert text == "The answer is 42."

    def test_responses_api_output_text(self):
        response = MagicMock()
        response.choices = None  # no choices
        response.output_text = "Response API answer."

        text = RAGPipeline._extract_text_from_response(response)
        assert text == "Response API answer."

    def test_empty_response(self):
        response = MagicMock(spec=[])  # no attributes
        text = RAGPipeline._extract_text_from_response(response)
        assert text == ""


# ---------------------------------------------------------------------------
# is_indexed / index & load round-trip
# ---------------------------------------------------------------------------
class TestIsIndexed:
    def test_false_when_empty(self, pipeline):
        assert pipeline.is_indexed() is False

    def test_true_after_files_exist(self, pipeline):
        import faiss
        # Create fake index and chunks files
        idx = faiss.IndexFlatL2(384)
        idx.add(np.random.rand(2, 384).astype("float32"))
        faiss.write_index(idx, os.path.join(pipeline.store_path, "faiss.index"))
        with open(os.path.join(pipeline.store_path, "chunks.json"), "w") as f:
            json.dump([{"content": "a", "metadata": {}}], f)

        assert pipeline.is_indexed() is True


class TestIndexAndLoad:
    def test_index_documents_creates_files(self, pipeline, sample_md_path):
        pipeline.index_documents([sample_md_path])

        assert os.path.exists(os.path.join(pipeline.store_path, "faiss.index"))
        assert os.path.exists(os.path.join(pipeline.store_path, "chunks.json"))
        assert pipeline.index is not None
        assert len(pipeline.chunks) > 0

    def test_load_index_restores_state(self, pipeline, sample_md_path):
        pipeline.index_documents([sample_md_path])
        num_chunks = len(pipeline.chunks)

        # Reset in-memory state
        pipeline.index = None
        pipeline.bm25 = None
        pipeline.chunks = []

        pipeline.load_index()
        assert len(pipeline.chunks) == num_chunks
        assert pipeline.index is not None
        assert pipeline.bm25 is not None


# ---------------------------------------------------------------------------
# query (integration-level, mocked LLM)
# ---------------------------------------------------------------------------
class TestQuery:
    def test_query_returns_answer_and_citations(self, pipeline, sample_md_path):
        pipeline.index_documents([sample_md_path])

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The answer based on context."
        del mock_response.output_text
        pipeline.client.chat.completions.create.return_value = mock_response

        answer, citations = pipeline.query("What are the project goals?")

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(citations, list)
        assert len(citations) > 0

    def test_query_vector_search(self, pipeline, sample_md_path):
        pipeline.index_documents([sample_md_path])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Vector answer."
        del mock_response.output_text
        pipeline.client.chat.completions.create.return_value = mock_response

        answer, citations = pipeline.query("test", search_type="vector")
        assert len(answer) > 0

    def test_query_keyword_search(self, pipeline, sample_md_path):
        pipeline.index_documents([sample_md_path])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Keyword answer."
        del mock_response.output_text
        pipeline.client.chat.completions.create.return_value = mock_response

        answer, citations = pipeline.query("test", search_type="keyword")
        assert len(answer) > 0

    def test_get_config(self, pipeline):
        config = pipeline.get_config()
        assert "deployment_name" in config
        assert config["deployment_name"] == "gpt-4"
        assert config["api_mode"] == "chat_completions"
