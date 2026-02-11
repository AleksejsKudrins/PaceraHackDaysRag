import streamlit as st
import os
import json
from chunking_utils import DocumentChunker

# Premium UI Configuration
st.set_page_config(
    page_title="RAG Chunking Studio",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 10px;
    }
    .file-container {
        border: 2px dashed #4A90E2;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        background-color: white;
        transition: all 0.3s ease;
    }
    .file-container:hover {
        border-color: #357ABD;
        background-color: #f0f7ff;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }
    .chunk-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4A90E2;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("📄 RAG Chunking Studio")
    st.subheader("Process your documents into optimized chunks for RAG")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        chunk_size = st.slider("Chunk Size (characters)", 100, 2000, 700)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100)
        use_tokens = st.checkbox("Use Token-based splitting", value=False)
        
        st.divider()
        st.info("Files are processed in real-time. No data is stored permanently.")

    # Main Upload Section
    st.markdown("### Please, upload your files here")
    uploaded_files = st.file_uploader(
        "Drag and drop your files (PDF, Markdown)", 
        type=["pdf", "md"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks = []
        
        with st.status("Processing documents...", expanded=True) as status:
            for uploaded_file in uploaded_files:
                st.write(f"Processing: **{uploaded_file.name}**")
                
                try:
                    if uploaded_file.name.endswith(".pdf"):
                        docs = chunker.load_pdf(uploaded_file, filename=uploaded_file.name)
                        if not docs:
                            st.warning(f"  - No text found in {uploaded_file.name} (it might be scanned).")
                            continue
                        chunks = chunker.chunk_documents(docs, use_tokens=use_tokens)
                    elif uploaded_file.name.endswith(".md"):
                        docs = chunker.load_markdown(uploaded_file, filename=uploaded_file.name)
                        # For markdown, we use the header-aware splitter if content is available
                        content = docs[0]["content"]
                        metadata = docs[0]["metadata"]
                        chunks = chunker.chunk_markdown_with_headers(content, metadata)
                    else:
                        st.error(f"Unsupported file type: {uploaded_file.name}")
                        continue
                        
                    st.success(f"  - Generated {len(chunks)} chunks.")
                    all_chunks.extend(chunks)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            status.update(label="Processing Complete!", state="complete", expanded=False)

        if all_chunks:
            # Layout for results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("Total Chunks", len(all_chunks))
                
                # Download button for JSON
                json_string = json.dumps(all_chunks, indent=2)
                st.download_button(
                    label="📥 Download Chunks (JSON)",
                    file_name="chunks_output.json",
                    mime="application/json",
                    data=json_string,
                )

            with col2:
                # Show a preview of the metadata
                st.markdown("#### Meta-Data Preview")
                if len(all_chunks) > 0:
                    st.json(all_chunks[0]["metadata"])

            st.divider()
            st.markdown("### Chunk Preview")
            
            # Show top 5 chunks
            for i, chunk in enumerate(all_chunks[:5]):
                with st.container():
                    st.markdown(f"""
                        <div class="chunk-card">
                            <strong>Chunk {i}</strong> (Source: {chunk['metadata'].get('source')})<br>
                            <p style="font-size: 0.9em; color: #555;">{chunk['content'][:500]}...</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            if len(all_chunks) > 5:
                st.info(f"Showing 5 out of {len(all_chunks)} chunks. Download the JSON for full results.")

    else:
        # Empty state display
        st.markdown("""
            <div style="padding: 100px; text-align: center; color: #666; background: white; border-radius: 15px; border: 1px solid #ddd;">
                <p style="font-size: 1.2em;">Drag and drop files to get started</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
