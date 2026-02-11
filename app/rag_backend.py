import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from openai import AzureOpenAI
from openai import NotFoundError
from chunking_utils import DocumentChunker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _normalize_azure_endpoint(endpoint: str) -> str:
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return endpoint
    # Common misconfig: users paste an endpoint that already includes /openai
    # The AzureOpenAI client will append /openai internally.
    lowered = endpoint.lower()
    openai_idx = lowered.find("/openai")
    if openai_idx != -1:
        endpoint = endpoint[:openai_idx]
    return endpoint.rstrip("/")

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG Pipeline...")

        # Initialize embedding model (cached after first load)
        try:
            logger.info("Loading embedding model: all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}", exc_info=True)
            raise

        # Initialize Azure OpenAI client
        azure_endpoint = _normalize_azure_endpoint(os.getenv('AZURE_OPENAI_ENDPOINT'))
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        # Note: newer models (e.g., GPT-5 deployments) often require newer preview API versions.
        api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')
        deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        api_mode = (os.getenv('AZURE_OPENAI_API_MODE', 'chat_completions') or '').strip().lower()

        # Back-compat aliases
        if api_mode in {"chat", "chatcompletions", "chat-completions"}:
            api_mode = "chat_completions"
        if api_mode in {"response", "responses_api", "responses-api"}:
            api_mode = "responses"

        logger.info(
            "Azure OpenAI configuration - Endpoint: %s, Version: %s, Deployment: %s, Mode: %s",
            azure_endpoint,
            api_version,
            deployment_name,
            api_mode,
        )

        if not azure_endpoint or not api_key:
            logger.error("Missing Azure OpenAI credentials in environment variables")
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in environment")

        if not deployment_name:
            raise ValueError(
                "AZURE_OPENAI_DEPLOYMENT_NAME must be set to your Azure OpenAI deployment name "
                "(this is the deployment name in Azure AI Foundry/Studio, not necessarily the model family name)."
            )

        if api_mode not in {"chat_completions", "responses"}:
            raise ValueError(
                "AZURE_OPENAI_API_MODE must be either 'chat_completions' or 'responses'. "
                f"Got: {api_mode!r}"
            )

        try:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
            self.deployment_name = deployment_name
            self.api_mode = api_mode
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}", exc_info=True)
            raise

    def _call_llm(self, prompt: str):
        if self.api_mode == "responses":
            # The OpenAI Python SDK has evolved; support both param names.
            kwargs = {
                "model": self.deployment_name,
                "instructions": "You are a helpful assistant that answers questions based on provided context.",
                "input": prompt,
                "temperature": 0.7,
            }
            try:
                return self.client.responses.create(**kwargs, max_output_tokens=1024)
            except TypeError:
                return self.client.responses.create(**kwargs, max_tokens=1024)
        else:
            return self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.7,
            )

    @staticmethod
    def _extract_text_from_response(response) -> str:
        # chat.completions
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            return getattr(msg, "content", None) or ""

        # responses
        if hasattr(response, "output_text"):
            return response.output_text or ""

        # conservative fallback for older SDK shapes
        if hasattr(response, "output") and response.output:
            text_parts = []
            for item in response.output:
                content = getattr(item, "content", None)
                if not content:
                    continue
                for part in content:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        text_parts.append(part_text)
            return "\n".join(text_parts).strip()

        return ""

        # FAISS index and metadata
        self.index = None
        self.chunks = []
        self.store_path = 'vector_store'
        os.makedirs(self.store_path, exist_ok=True)
        logger.info(f"Vector store path: {self.store_path}")
        logger.info("RAG Pipeline initialization complete")

    def is_indexed(self) -> bool:
        """Check if vector store already exists"""
        index_exists = os.path.exists(f'{self.store_path}/faiss.index')
        chunks_exist = os.path.exists(f'{self.store_path}/chunks.json')
        is_indexed = index_exists and chunks_exist
        logger.info(f"Checking if indexed: index_exists={index_exists}, chunks_exist={chunks_exist}, result={is_indexed}")
        return is_indexed

    def index_documents(self, doc_paths: List[str]):
        """Chunk, embed, and store documents in vector database"""
        logger.info(f"Starting document indexing for {len(doc_paths)} documents: {doc_paths}")

        try:
            chunker = DocumentChunker(chunk_size=700, chunk_overlap=100)
            all_chunks = []

            # Process each document
            for path in doc_paths:
                logger.info(f"Processing document: {path}")
                try:
                    if path.endswith('.pdf'):
                        docs = chunker.load_pdf(path)
                        chunks = chunker.chunk_documents(docs)
                        logger.info(f"PDF processed: {len(chunks)} chunks created from {path}")
                    elif path.endswith('.md'):
                        docs = chunker.load_markdown(path)
                        chunks = chunker.chunk_markdown_with_headers(
                            docs[0]['content'],
                            docs[0]['metadata']
                        )
                        logger.info(f"Markdown processed: {len(chunks)} chunks created from {path}")
                    else:
                        logger.warning(f"Skipping unsupported file type: {path}")
                        continue
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing document {path}: {str(e)}", exc_info=True)
                    continue

            logger.info(f"Total chunks created: {len(all_chunks)}")

            # Generate embeddings (batch process)
            logger.info("Generating embeddings...")
            texts = [chunk['content'] for chunk in all_chunks]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            logger.info(f"Embeddings generated: shape={embeddings.shape}")

            # Create FAISS index
            dimension = embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension={dimension}")
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            self.chunks = all_chunks
            logger.info(f"FAISS index created with {self.index.ntotal} vectors")

            # Persist to disk
            index_path = f'{self.store_path}/faiss.index'
            chunks_path = f'{self.store_path}/chunks.json'
            logger.info(f"Saving index to {index_path}")
            faiss.write_index(self.index, index_path)
            logger.info(f"Saving chunks metadata to {chunks_path}")
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, indent=2)
            logger.info("Document indexing complete and saved to disk")

        except Exception as e:
            logger.error(f"Critical error during document indexing: {str(e)}", exc_info=True)
            raise

    def load_index(self):
        """Load existing vector store from disk"""
        try:
            index_path = f'{self.store_path}/faiss.index'
            chunks_path = f'{self.store_path}/chunks.json'

            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")

            logger.info(f"Loading chunks metadata from {chunks_path}")
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            logger.info(f"Loaded {len(self.chunks)} chunks metadata")

            # Get file sizes for debugging
            index_size = os.path.getsize(index_path) / (1024 * 1024)  # MB
            chunks_size = os.path.getsize(chunks_path) / (1024 * 1024)  # MB
            logger.info(f"Index file size: {index_size:.2f} MB, Chunks file size: {chunks_size:.2f} MB")

        except FileNotFoundError as e:
            logger.error(f"Vector store files not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}", exc_info=True)
            raise

    def query(self, question: str, k: int = 5) -> Tuple[str, List[str]]:
        """Retrieve relevant chunks and generate answer with citations"""
        logger.info(f"Processing query: '{question}' (k={k})")

        try:
            # Load index if not already loaded
            if self.index is None:
                logger.info("Index not loaded, loading from disk...")
                self.load_index()

            # Embed the question
            logger.info("Embedding query...")
            query_embedding = self.embedding_model.encode([question])[0]
            logger.info(f"Query embedded: shape={query_embedding.shape}")

            # Search for top-k similar chunks
            logger.info(f"Searching FAISS index for top-{k} similar chunks...")
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                k
            )
            logger.info(f"FAISS search complete. Distances: {distances[0]}, Indices: {indices[0]}")

            # Retrieve chunks
            retrieved_chunks = [self.chunks[i] for i in indices[0]]
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

            # Build context for LLM
            context = "\n\n".join([
                f"[Source: {chunk['metadata']['source']}]\n{chunk['content']}"
                for chunk in retrieved_chunks
            ])
            logger.info(f"Context built: {len(context)} characters from {len(retrieved_chunks)} chunks")

            # Construct prompt
            prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say so.

Context:
---
{context}
---

Question: {question}

Answer:"""

            # Call Azure OpenAI API
            logger.info(
                "Calling Azure OpenAI API (deployment=%s, mode=%s)...",
                self.deployment_name,
                self.api_mode,
            )
            try:
                response = self._call_llm(prompt)
                logger.info("Azure OpenAI API call successful")
            except NotFoundError as e:
                logger.error("Azure OpenAI API call failed (404 Not Found)")
                logger.error(
                    "Troubleshooting 404: verify AZURE_OPENAI_DEPLOYMENT_NAME exists; verify AZURE_OPENAI_ENDPOINT is like https://<resource>.openai.azure.com (no /openai); "
                    "verify AZURE_OPENAI_API_VERSION supports your deployment; if using GPT-5 deployments, set AZURE_OPENAI_API_MODE=responses.")
                logger.error(f"Azure OpenAI details - deployment={self.deployment_name}, mode={self.api_mode}")
                raise
            except Exception as e:
                logger.error(f"Azure OpenAI API call failed: {str(e)}", exc_info=True)
                raise

            answer = self._extract_text_from_response(response)
            logger.info(f"Generated answer: {len(answer)} characters")

            # Extract unique citations
            citations = []
            seen_sources = set()
            for chunk in retrieved_chunks:
                source = chunk['metadata']['source']
                if source not in seen_sources:
                    citation = f"Source: {source}"
                    if 'page' in chunk['metadata']:
                        citation += f" (Page {chunk['metadata']['page']})"
                    elif 'Header 1' in chunk['metadata']:
                        citation += f" (Section: {chunk['metadata']['Header 1']})"
                    citations.append(citation)
                    seen_sources.add(source)

            logger.info(f"Extracted {len(citations)} unique citations")
            logger.info(f"Query complete: question='{question}', answer_length={len(answer)}, citations={len(citations)}")

            return answer, citations

        except Exception as e:
            logger.error(f"Error processing query '{question}': {str(e)}", exc_info=True)
            raise
