import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from openai import AzureOpenAI
from chunking_utils import DocumentChunker

class RAGPipeline:
    def __init__(self):
        # Initialize embedding model (cached after first load)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Azure OpenAI client
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4')

        if not azure_endpoint or not api_key:
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in environment")

        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name

        # FAISS index and metadata
        self.index = None
        self.chunks = []
        self.store_path = 'vector_store'
        os.makedirs(self.store_path, exist_ok=True)

    def is_indexed(self) -> bool:
        """Check if vector store already exists"""
        return (os.path.exists(f'{self.store_path}/faiss.index') and
                os.path.exists(f'{self.store_path}/chunks.json'))

    def index_documents(self, doc_paths: List[str]):
        """Chunk, embed, and store documents in vector database"""
        chunker = DocumentChunker(chunk_size=700, chunk_overlap=100)
        all_chunks = []

        # Process each document
        for path in doc_paths:
            if path.endswith('.pdf'):
                docs = chunker.load_pdf(path)
                chunks = chunker.chunk_documents(docs)
            elif path.endswith('.md'):
                docs = chunker.load_markdown(path)
                chunks = chunker.chunk_markdown_with_headers(
                    docs[0]['content'],
                    docs[0]['metadata']
                )
            else:
                continue
            all_chunks.extend(chunks)

        # Generate embeddings (batch process)
        texts = [chunk['content'] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        self.chunks = all_chunks

        # Persist to disk
        faiss.write_index(self.index, f'{self.store_path}/faiss.index')
        with open(f'{self.store_path}/chunks.json', 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2)

    def load_index(self):
        """Load existing vector store from disk"""
        self.index = faiss.read_index(f'{self.store_path}/faiss.index')
        with open(f'{self.store_path}/chunks.json', 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

    def query(self, question: str, k: int = 5) -> Tuple[str, List[str]]:
        """Retrieve relevant chunks and generate answer with citations"""
        # Load index if not already loaded
        if self.index is None:
            self.load_index()

        # Embed the question
        query_embedding = self.embedding_model.encode([question])[0]

        # Search for top-k similar chunks
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )

        # Retrieve chunks
        retrieved_chunks = [self.chunks[i] for i in indices[0]]

        # Build context for LLM
        context = "\n\n".join([
            f"[Source: {chunk['metadata']['source']}]\n{chunk['content']}"
            for chunk in retrieved_chunks
        ])

        # Construct prompt
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say so.

Context:
---
{context}
---

Question: {question}

Answer:"""

        # Call Azure OpenAI API
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )

        answer = response.choices[0].message.content

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

        return answer, citations
