# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system built during a hackathon. The project consists of:
1. A Streamlit-based UI that provides an interactive RAG chat interface ([app/main.py](app/main.py))
2. Document chunking utilities for processing PDFs and Markdown files ([app/chunking_utils.py](app/chunking_utils.py))
3. A complete RAG pipeline with FAISS vector database and Azure OpenAI integration ([app/rag_backend.py](app/rag_backend.py))

**Current State:** Fully functional RAG system with semantic document retrieval, vector embeddings (sentence-transformers), FAISS vector database, and Azure OpenAI for answer generation.

## Development Commands

### Running the Streamlit UI
```bash
# Using Docker (recommended)
docker-compose up --build

# Access at http://localhost:8501
```

The Dockerfile references [app/requirements.txt](app/requirements.txt) and runs [app/main.py](app/main.py) via Streamlit.

### Working with Chunking Utilities
```bash
cd "Python chunk creation logic"

# Install dependencies
pip install -r requirements.txt

# Run the chunking demo
python demo_chunking.py

# Debug PDF extraction
python debug_pdf.py
```

## Project Structure

### Two Separate Codebases

The repository contains two disconnected components:

**1. Streamlit UI (`app/`):**
- [main.py](app/main.py): Mock RAG chat interface with streaming responses
- Uses Streamlit session state for conversation history
- Returns dummy citations (hardcoded source references)
- No actual RAG pipeline implementation yet

**2. Chunking Logic (`Python chunk creation logic/`):**
- [chunking_utils.py](chunking_utils.py): Core document processing
  - `DocumentChunker`: Main class with configurable chunk size/overlap
  - `load_pdf()`: Extracts text from PDFs using pypdf
  - `load_markdown()`: Loads Markdown files
  - `chunk_documents()`: Splits documents using RecursiveCharacterTextSplitter or TokenTextSplitter
  - `chunk_markdown_with_headers()`: Header-aware Markdown splitting that preserves document structure
- [demo_chunking.py](demo_chunking.py): Example usage that processes docs and outputs JSON
- Uses LangChain text splitters (RecursiveCharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter)

### Key Files

- [docker-compose.yml](docker-compose.yml): Defines the `rag-app` service on port 8501
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml): Automated deployment to a Linux server via SSH
- [docs/rag-project-overview.md](docs/rag-project-overview.md): Project goals, success criteria, and learning objectives
- [Python chunk creation logic/Agent comparison.md](Python chunk creation logic/Agent comparison.md): Notes on AI agent testing

## Architecture Notes

### Document Chunking Strategy

The chunking utilities support multiple approaches:
- **RecursiveCharacterTextSplitter** (default): Splits on paragraphs → sentences → words for better semantic coherence
- **TokenTextSplitter**: Precise token-based splitting for LLM context window management
- **Header-aware splitting**: Preserves Markdown structure by splitting on headers first, then subdividing large sections

Metadata preservation is critical: each chunk carries source file, page number (for PDFs), chunk ID, and header context (for Markdown).

### RAG Pipeline Architecture

The system implements a complete RAG pipeline with the following components:

**1. Vector Embeddings**: Uses sentence-transformers (all-MiniLM-L6-v2) for generating 384-dimensional embeddings
- Free and local (no API costs)
- Fast embedding generation (~5-10s for 1000 chunks)
- Suitable for semantic similarity search

**2. Vector Database**: FAISS (Facebook AI Similarity Search)
- File-based persistence (`app/vector_store/faiss.index`)
- Fast similarity search (~10ms for 1000 chunks)
- IndexFlatL2 for exact nearest neighbor search

**3. LLM Integration**: Azure OpenAI
- Configured via environment variables (endpoint, API key, deployment name)
- Uses GPT-4 (or configured deployment) for answer generation
- Prompt engineering to ensure answers stay grounded in retrieved context

**4. Data Flow**:
```
User Question → Embed Query → FAISS Search → Top-5 Chunks →
Build Context → Azure OpenAI → Extract Answer + Citations → Display
```

**5. Citation Tracking**: Automatically extracts source file names, page numbers (PDFs), and section headers (Markdown) from retrieved chunk metadata

## Missing Integration

While the core RAG pipeline is functional, these enhancements would improve the system:

### 1. **File Upload Integration with Vector Store**
**Current State**: The UI has file upload functionality (sidebar), but uploaded documents are not automatically added to the FAISS vector store.

**What's Missing**:
- After user uploads and processes files, chunks should be embedded and added to the existing FAISS index
- Need `add_documents()` method in `RAGPipeline` to dynamically extend the vector store
- Uploaded documents should persist across sessions (currently only in `st.session_state`)

**Implementation Path**:
```python
# In rag_backend.py
def add_new_documents(self, new_chunks: List[Dict]):
    """Add newly uploaded document chunks to existing vector store"""
    texts = [chunk['content'] for chunk in new_chunks]
    embeddings = self.embedding_model.encode(texts)
    self.index.add(embeddings.astype('float32'))
    self.chunks.extend(new_chunks)
    # Save updated index
    faiss.write_index(self.index, f'{self.store_path}/faiss.index')
    # Save updated chunks metadata
```

### 2. **Hybrid Search (Vector + Keyword)**
**Current State**: Only semantic vector search via FAISS

**What's Missing**:
- BM25 or TF-IDF keyword search for exact term matching
- Score fusion (e.g., RRF - Reciprocal Rank Fusion) to combine vector and keyword results
- Better handling of queries with specific technical terms or entity names

**Why Needed**: Pure vector search can miss exact keyword matches that users expect

### 3. **Re-ranking**
**Current State**: Returns top-5 chunks directly from FAISS

**What's Missing**:
- Cross-encoder model for re-scoring retrieved chunks
- Would improve precision by reordering FAISS results based on query-chunk relevance

**Implementation**: Use `sentence-transformers/ms-marco-MiniLM-L-6-v2` cross-encoder

### 4. **Query Expansion & Refinement**
**What's Missing**:
- Query rewriting (e.g., "What are success metrics?" → "success criteria, evaluation metrics, performance targets")
- Synonym expansion for better recall
- Multi-query retrieval (generate multiple query variants, retrieve for each, merge results)

### 5. **Conversation Memory**
**Current State**: Each query is independent

**What's Missing**:
- Multi-turn conversation context (refer back to previous Q&A)
- Session-based chat history fed into LLM prompts
- Follow-up question handling ("What else?", "Tell me more")

### 6. **Evaluation & Monitoring**
**What's Missing**:
- Retrieval quality metrics (precision@k, recall@k, MRR)
- Answer quality tracking (faithfulness, relevance)
- Query latency monitoring
- User feedback mechanism (thumbs up/down on answers)

### 7. **Advanced Chunking Strategies**
**Current State**: Fixed-size chunks with overlap

**What's Missing**:
- Semantic chunking (split on topic boundaries)
- Hierarchical chunking (parent-child relationships)
- Overlap optimization based on document type
- Table and code-specific chunking

### 8. **Document Management UI**
**What's Missing**:
- View all indexed documents
- Delete documents from vector store
- Re-index specific documents
- Bulk upload via folder
- Document status (indexed/pending/failed)

### 9. **Authentication & Access Control**
**What's Missing**:
- User authentication (currently publicly accessible)
- Document-level permissions
- Query logging per user
- Admin dashboard

### 10. **Testing**
**Current State**: No tests exist

**What's Missing**:
- Unit tests for chunking logic
- Integration tests for RAG pipeline
- End-to-end UI tests
- Retrieval quality evaluation suite

## Deployment

The project uses GitHub Actions to deploy to a remote Linux server:
- Workflow triggers on push to `main` or manual dispatch
- SSH deployment to `/opt/rag-app` on the target server
- Automatically installs Docker if missing
- Runs `docker compose up -d --build` to deploy

**Required GitHub Secrets**:
- `SERVER_IP`, `SERVER_USER`, `SSH_PRIVATE_KEY` (for SSH deployment)
- `AZURE_OPENAI_ENDPOINT` (e.g., https://your-resource.openai.azure.com/)
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION` (e.g., 2025-01-01-preview)
- `AZURE_OPENAI_DEPLOYMENT_NAME` (e.g., gpt-4)
- `AZURE_OPENAI_API_MODE` (optional: `chat_completions` or `responses`)
- `CLOUDFLARED_TOKEN` (optional, for Cloudflare tunnel)

## Dependencies

**Core Application** ([app/requirements.txt](app/requirements.txt)):
- `streamlit` - Web UI framework
- `sentence-transformers` - Local embedding model (all-MiniLM-L6-v2)
- `faiss-cpu` - Vector database for similarity search
- `openai` - Azure OpenAI SDK for LLM integration
- `langchain-text-splitters` - Document chunking strategies
- `pypdf` - PDF text extraction
- `tiktoken` - Token counting for chunk sizing

**Environment Variables** (`.env` file):
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI resource endpoint
- `AZURE_OPENAI_API_KEY` - API key from Azure Portal
- `AZURE_OPENAI_API_VERSION` - API version (default: 2025-01-01-preview)
- `AZURE_OPENAI_DEPLOYMENT_NAME` - Your deployed model name (e.g., gpt-4)
- `AZURE_OPENAI_API_MODE` - Optional: `chat_completions` (default) or `responses` (use `responses` for GPT-5 deployments)
- `CLOUDFLARED_TOKEN` - Optional Cloudflare tunnel token

## Project Goals

Per [docs/rag-project-overview.md](docs/rag-project-overview.md):
- Demonstrate semantic document retrieval to reduce LLM context size and cost
- Learn document chunking strategies, vector embeddings, and prompt engineering
- Success metrics: 90%+ retrieval relevance, <5s query latency, source traceability

## Working in this Repository

- The main application is in [app/](app/) - includes UI, RAG backend, and chunking utilities
  - [main.py](app/main.py) - Streamlit UI with RAG integration
  - [rag_backend.py](app/rag_backend.py) - RAG pipeline (embeddings, FAISS, Azure OpenAI)
  - [chunking_utils.py](app/chunking_utils.py) - Document processing utilities
- [Python chunk creation logic/](Python chunk creation logic/) - Standalone demos and experiments
- Vector store persists in `app/vector_store/` (git-ignored, mounted as Docker volume)
- No tests exist yet
- The project is fully containerized - `docker-compose up --build` runs everything

**First-time setup**:
1. Copy `.env.example` to `.env` and fill in Azure OpenAI credentials
2. Run `docker-compose up --build`
3. Wait for model download (~80MB) and initial document indexing (30-60s)
4. Access at http://localhost:8501
