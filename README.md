# RAG Demo - Smart Document Retrieval for LLMs

A **Retrieval-Augmented Generation (RAG)** application that enables intelligent question-answering over document collections using vector embeddings and large language models.

## 🎯 Overview

This project solves the challenge of querying large document collections efficiently. Instead of sending entire document corpuses to a language model (which is slow, expensive, and often impossible due to context limits), this RAG pipeline:

- **Indexes** documents by chunking them and generating vector embeddings
- **Retrieves** only the most relevant document chunks for each query using semantic search
- **Generates** accurate, grounded answers with source citations
- **Reduces** costs and hallucinations by providing focused context to the LLM

## ✨ Features

- 📄 **Multi-format document support** (PDF, Markdown, and more)
- 🔍 **Semantic search** using FAISS vector database
- 🤖 **Azure OpenAI integration** for answer generation
- 💬 **Interactive Streamlit UI** with chat interface
- 📚 **Knowledge base management** - upload and index new documents
- 📊 **Source citations** - every answer includes document references
- 🐳 **Docker deployment** with Cloudflare Tunnel support
- 🚀 **CI/CD pipeline** with GitHub Actions

## 🏗️ Architecture

```
User Query → Vector Search (FAISS) → Top-K Relevant Chunks → LLM (Azure OpenAI) → Answer + Citations
```

1. **Document Ingestion**: Documents are chunked using semantic splitting strategies
2. **Embedding Generation**: Chunks are converted to vector embeddings using sentence-transformers
3. **Vector Storage**: Embeddings are stored in FAISS for fast similarity search
4. **Query Processing**: User questions are embedded and matched against the vector database
5. **Answer Generation**: Retrieved chunks are sent to Azure OpenAI as context for answering

## 🛠️ Technologies

- **Python 3.9+**
- **Streamlit** - Interactive web interface
- **FAISS** - Vector database for similarity search
- **Sentence Transformers** - Embedding model
- **Azure OpenAI** - Language model for generation
- **LangChain** - Text splitting utilities
- **Docker & Docker Compose** - Containerization
- **Cloudflare Tunnel** - Secure public access

## 📋 Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- Azure OpenAI API access with a deployed model

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PaceraHackDaysRag
   ```

2. **Install dependencies**
   ```bash
   cd app
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
   ```

4. **Run the application**
   ```bash
   streamlit run app/main.py
   ```
   
   Access the app at `http://localhost:8501`

### Docker Deployment

1. **Configure environment variables**
   
   Create a `.env` file with your Azure OpenAI credentials and optional Cloudflare token:
   ```env
   AZURE_OPENAI_ENDPOINT=your-endpoint
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment
   CLOUDFLARED_TOKEN=your-cloudflare-token  # Optional
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker compose up --build
   ```
   
   The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
PaceraHackDaysRag/
├── app/
│   ├── main.py                 # Streamlit application entry point
│   ├── rag_backend.py         # RAG pipeline implementation
│   ├── chunking_utils.py      # Document processing and chunking
│   ├── requirements.txt       # Python dependencies
│   └── vector_store/          # FAISS index storage (created at runtime)
├── docs/
│   ├── rag-project-overview.md    # Project documentation
│   └── team-ai-usage.md           # Team guidelines
├── Python chunk creation logic/   # Experimental chunking implementations
│   ├── app_upload.py
│   ├── chunking_utils.py
│   ├── demo_chunking.py
│   └── debug_pdf.py
├── .github/
│   └── workflows/
│       └── deploy.yml         # CI/CD deployment workflow
├── docker-compose.yml         # Docker services configuration
├── Dockerfile                 # Application container definition
├── DEPLOYMENT.md             # Deployment instructions
└── README.md                 # This file
```

## 🎮 Usage

### Uploading Documents

1. Launch the application
2. Use the **sidebar** to upload documents (PDF, Markdown, etc.)
3. Click **"Index Uploaded Files"** to process and add them to the knowledge base
4. View processed chunks in the expandable section

### Asking Questions

1. Type your question in the chat input at the bottom
2. The system will:
   - Search for the most relevant document chunks
   - Send them with your query to Azure OpenAI
   - Display the answer with source citations
3. View the conversation history in the chat interface

### Managing the Knowledge Base

- **View Status**: Check the sidebar for indexed document count
- **Add Documents**: Upload new files and reindex
- **Persistent Storage**: The vector store is saved to disk and loaded on startup

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_API_VERSION` | API version (e.g., 2024-02-15-preview) | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Name of your deployed model | Yes |
| `CLOUDFLARED_TOKEN` | Cloudflare Tunnel token for public access | No |

### Chunking Configuration

Adjust in [chunking_utils.py](app/chunking_utils.py):
- `chunk_size`: Maximum characters per chunk (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- Splitting strategy: Semantic-based or fixed-size

### RAG Parameters

Adjust in [rag_backend.py](app/rag_backend.py):
- `k`: Number of chunks to retrieve (default: 5)
- Embedding model: sentence-transformers model selection
- Prompt template: Customize the system prompt

## 🚢 Deployment

### Production Deployment

The project includes automated deployment via GitHub Actions. See [DEPLOYMENT.md](DEPLOYMENT.md) for details.

**Required GitHub Secrets:**
- `SERVER_IP` - Target server address
- `SERVER_USER` - SSH username
- `SSH_PRIVATE_KEY` - SSH private key for authentication

Push to the `main` branch to trigger automatic deployment.

### Cloudflare Tunnel

To expose the application publicly with Cloudflare Tunnel:
1. Create a tunnel at [Cloudflare Zero Trust](https://one.dash.cloudflare.com/)
2. Add the tunnel token to your `.env` file
3. The Docker Compose setup includes the cloudflared service

## 📊 Success Metrics

| Metric | Target |
|--------|--------|
| Retrieval relevance | Top-5 chunks contain correct answer ≥90% |
| Answer accuracy | Factually correct and grounded ≥85% |
| Context efficiency | Only 3-10 relevant chunks sent per query |
| Latency | End-to-end response < 5 seconds |
| Scalability | Handles 1,000+ documents without degradation |

## 🤝 Contributing

This project was developed during an AI Hack Day focused on intelligent search and RAG systems. Contributions and improvements are welcome!

### Areas for Enhancement

- [ ] Support for additional document formats (Word, HTML, etc.)
- [ ] Advanced chunking strategies (sliding window, semantic-aware)
- [ ] Multiple embedding model support
- [ ] Query refinement and multi-turn conversations
- [ ] Evaluation metrics dashboard
- [ ] Hybrid search (combining vector and keyword search)
- [ ] Multi-language support

## 📝 License

[Specify your license here]

## 👥 Team

Developed during Pacera AI Hack Days 2026

## 🔗 Resources

- [RAG Project Overview](docs/rag-project-overview.md)
- [Team AI Usage Guidelines](docs/team-ai-usage.md)
- [Deployment Guide](DEPLOYMENT.md)
