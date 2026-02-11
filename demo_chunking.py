import os
import json
from chunking_utils import DocumentChunker

def main():
    # Initialize chunker with specific size and overlap
    # Smaller chunks are often better for precision, larger for context.
    # 500-1000 characters is a common starting point.
    chunker = DocumentChunker(chunk_size=700, chunk_overlap=100)
    
    files_to_process = [
        os.path.join("..", "..", "docs", "rag-project-overview.md"),
        os.path.join("..", "..", "research", "RAG_Mastery_Building_Dynamic_AI.pdf")
    ]
    
    all_chunks = []
    
    print("--- RAG Chunking Demonstration ---\n")
    
    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
            
        print(f"Processing: {file_path}...")
        
        if file_path.endswith(".pdf"):
            docs = chunker.load_pdf(file_path)
            print(f"  - Loaded {len(docs)} pages from PDF.")
            if len(docs) == 0:
                print("    (Note: PDF may be scanned/image-only, skipping further processing for now)")
            chunks = chunker.chunk_documents(docs)
        elif file_path.endswith(".md"):
            docs = chunker.load_markdown(file_path)
            # Use the header-aware splitter for better context
            chunks = chunker.chunk_markdown_with_headers(docs[0]["content"], docs[0]["metadata"])
        else:
            print(f"Unsupported file type: {file_path}")
            continue
        print(f"  - Generated {len(chunks)} chunks.")
        all_chunks.extend(chunks)

    # Summarize findings
    print(f"\nTotal chunks generated: {len(all_chunks)}")
    
    # Save results to a file for inspection
    output_file = "chunks_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Show a few examples of metadata preservation
    if all_chunks:
        print("\n--- Example Chunk Metadata ---")
        for i in [0, len(all_chunks) // 2, len(all_chunks) - 1]:
            if i < len(all_chunks):
                chunk = all_chunks[i]
                print(f"\nChunk {i}:")
                print(f"  Source: {chunk['metadata'].get('source')}")
                print(f"  Type: {chunk['metadata'].get('type')}")
                if 'page' in chunk['metadata']:
                    print(f"  Page: {chunk['metadata'].get('page')}")
                print(f"  Content Snippet: {chunk['content'][:150].replace('\\n', ' ')}...")

if __name__ == "__main__":
    main()
