import io
import pytest
from chunking_utils import DocumentChunker


class TestLoadMarkdown:
    def test_load_from_file_path(self, sample_md_path):
        chunker = DocumentChunker()
        docs = chunker.load_markdown(sample_md_path)

        assert len(docs) == 1
        assert "Project Overview" in docs[0]["content"]
        assert docs[0]["metadata"]["source"] == "sample.md"
        assert docs[0]["metadata"]["type"] == "markdown"

    def test_load_from_file_like_object(self, sample_md_content):
        chunker = DocumentChunker()
        file_obj = io.BytesIO(sample_md_content.encode("utf-8"))
        docs = chunker.load_markdown(file_obj, filename="uploaded.md")

        assert len(docs) == 1
        assert docs[0]["metadata"]["source"] == "uploaded.md"
        assert "Project Overview" in docs[0]["content"]

    def test_default_filename_for_file_object(self):
        chunker = DocumentChunker()
        file_obj = io.BytesIO(b"# Hello")
        docs = chunker.load_markdown(file_obj)

        assert docs[0]["metadata"]["source"] == "uploaded_file.md"

    def test_explicit_filename_overrides_path(self, sample_md_path):
        chunker = DocumentChunker()
        docs = chunker.load_markdown(sample_md_path, filename="custom_name.md")

        assert docs[0]["metadata"]["source"] == "custom_name.md"


class TestLoadPdf:
    def test_load_from_file_path(self, sample_pdf_path):
        chunker = DocumentChunker()
        docs = chunker.load_pdf(sample_pdf_path)

        assert len(docs) == 2
        assert "page one" in docs[0]["content"].lower()
        assert docs[0]["metadata"]["source"] == "sample.pdf"
        assert docs[0]["metadata"]["page"] == 1
        assert docs[0]["metadata"]["type"] == "pdf"
        assert docs[1]["metadata"]["page"] == 2

    def test_explicit_filename(self, sample_pdf_path):
        chunker = DocumentChunker()
        docs = chunker.load_pdf(sample_pdf_path, filename="report.pdf")

        assert docs[0]["metadata"]["source"] == "report.pdf"


class TestChunkDocuments:
    def test_splits_large_documents(self, sample_documents):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents(sample_documents)

        # Each document is >500 chars, so both should be split
        assert len(chunks) > 2

    def test_preserves_metadata(self, sample_documents):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents(sample_documents)

        for chunk in chunks:
            assert "source" in chunk["metadata"]
            assert "type" in chunk["metadata"]
            assert "chunk_id" in chunk["metadata"]

    def test_chunk_ids_are_sequential_per_document(self, sample_documents):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents(sample_documents)

        doc1_chunks = [c for c in chunks if c["metadata"]["source"] == "doc1.md"]
        doc1_ids = [c["metadata"]["chunk_id"] for c in doc1_chunks]
        assert doc1_ids == list(range(len(doc1_ids)))

    def test_small_document_produces_single_chunk(self):
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        docs = [{"content": "Short text.", "metadata": {"source": "s.md", "type": "markdown"}}]
        chunks = chunker.chunk_documents(docs)

        assert len(chunks) == 1
        assert chunks[0]["content"] == "Short text."
        assert chunks[0]["metadata"]["chunk_id"] == 0

    def test_token_splitter_mode(self, sample_documents):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_documents(sample_documents, use_tokens=True)

        assert len(chunks) > 2
        for chunk in chunks:
            assert "chunk_id" in chunk["metadata"]


class TestChunkMarkdownWithHeaders:
    def test_produces_chunks_with_header_metadata(self, sample_md_content):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        metadata_base = {"source": "sample.md", "type": "markdown"}
        chunks = chunker.chunk_markdown_with_headers(sample_md_content, metadata_base)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "source" in chunk["metadata"]
            assert "sub_chunk_id" in chunk["metadata"]

    def test_preserves_base_metadata(self, sample_md_content):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        metadata_base = {"source": "test.md", "type": "markdown", "custom": "value"}
        chunks = chunker.chunk_markdown_with_headers(sample_md_content, metadata_base)

        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test.md"
            assert chunk["metadata"]["custom"] == "value"

    def test_large_section_gets_sub_chunked(self):
        # Build a markdown doc with one very large section
        large_section = "Some content. " * 200  # ~2800 chars
        markdown = f"# Big Section\n\n{large_section}"

        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_markdown_with_headers(markdown, {"source": "big.md"})

        # Should produce multiple sub-chunks
        assert len(chunks) > 1
        sub_ids = [c["metadata"]["sub_chunk_id"] for c in chunks]
        assert max(sub_ids) > 0
