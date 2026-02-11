import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from openai import AzureOpenAI
from chunking_utils import DocumentChunker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4')

        logger.info(f"Azure OpenAI configuration - Endpoint: {azure_endpoint}, Version: {api_version}, Deployment: {deployment_name}")

        if not azure_endpoint or not api_key:
            logger.error("Missing Azure OpenAI credentials in environment variables")
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in environment")

        try:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
            self.deployment_name = deployment_name
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}", exc_info=True)
            raise

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
            logger.info(f"Calling Azure OpenAI API (model={self.deployment_name})...")
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,
                    temperature=0.7
                )
                logger.info("Azure OpenAI API call successful")
            except Exception as e:
                logger.error(f"Azure OpenAI API call failed: {str(e)}", exc_info=True)
                raise

            answer = response.choices[0].message.content
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
