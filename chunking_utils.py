import os
from typing import List, Dict
from pypdf import PdfReader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, 
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter is generally the best all-around splitter.
        # It tries to split on paragraphs, then sentences, then words.
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        
        # TokenTextSplitter is useful when you want to be precise about token counts (important for LLMs)
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_pdf(self, file_path: str) -> List[Dict]:
        """Loads a PDF and returns a list of page contents with metadata."""
        documents = []
        reader = PdfReader(file_path)
        file_name = os.path.basename(file_path)
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append({
                    "content": text,
                    "metadata": {
                        "source": file_name,
                        "page": i + 1,
                        "type": "pdf"
                    }
                })
        return documents

    def load_markdown(self, file_path: str) -> List[Dict]:
        """Loads a Markdown file and returns its content with metadata."""
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        return [{
            "content": text,
            "metadata": {
                "source": file_name,
                "type": "markdown"
            }
        }]

    def chunk_documents(self, documents: List[Dict], use_tokens: bool = False) -> List[Dict]:
        """Chunks a list of documents while preserving metadata."""
        splitter = self.token_splitter if use_tokens else self.recursive_splitter
        
        chunks = []
        for doc in documents:
            texts = splitter.split_text(doc["content"])
            for i, text in enumerate(texts):
                # Copy original metadata and add chunk-specific info
                chunk_metadata = doc["metadata"].copy()
                chunk_metadata["chunk_id"] = i
                
                chunks.append({
                    "content": text,
                    "metadata": chunk_metadata
                })
        return chunks

    def chunk_markdown_with_headers(self, markdown_text: str, metadata_base: Dict) -> List[Dict]:
        """
        Splits markdown based on headers and then further splits large sections.
        This preserves the structural context of the document.
        """
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_sections = header_splitter.split_text(markdown_text)
        
        chunks = []
        for i, section in enumerate(header_sections):
            # Combined metadata: Base + Headers from splitter
            section_metadata = metadata_base.copy()
            section_metadata.update(section.metadata)
            
            # If section is too large, split it further using recursive splitter
            if len(section.page_content) > self.chunk_size:
                sub_chunks = self.recursive_splitter.split_text(section.page_content)
                for j, sub_content in enumerate(sub_chunks):
                    final_metadata = section_metadata.copy()
                    final_metadata["sub_chunk_id"] = j
                    chunks.append({
                        "content": sub_content,
                        "metadata": final_metadata
                    })
            else:
                section_metadata["sub_chunk_id"] = 0
                chunks.append({
                    "content": section.page_content,
                    "metadata": section_metadata
                })
        return chunks

if __name__ == "__main__":
    # Example usage
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    
    # Test with a local markdown file if it exists
    test_file = "rag-project-overview.md"
    if os.path.exists(test_file):
        docs = chunker.load_markdown(test_file)
        chunks = chunker.chunk_documents(docs)
        print(f"Total chunks for {test_file}: {len(chunks)}")
        if chunks:
            print("\nFirst chunk preview:")
            print(f"Metadata: {chunks[0]['metadata']}")
            print(f"Content snippet: {chunks[0]['content'][:200]}...")
