import os
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def sample_md_path():
    return os.path.join(FIXTURES_DIR, "sample.md")


@pytest.fixture
def sample_pdf_path():
    return os.path.join(FIXTURES_DIR, "sample.pdf")


@pytest.fixture
def sample_md_content(sample_md_path):
    with open(sample_md_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def sample_documents():
    """A list of pre-built document dicts for testing chunk_documents()."""
    return [
        {
            "content": "First document. " * 100,  # ~1600 chars
            "metadata": {"source": "doc1.md", "type": "markdown"},
        },
        {
            "content": "Second document content. " * 50,  # ~1250 chars
            "metadata": {"source": "doc2.md", "type": "markdown"},
        },
    ]
