import streamlit as st
import time
import sys
import os

# Add the parent directory to sys.path so we can import chunking_utils
# if we are running from the app folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chunking_utils import DocumentChunker

# Configure the Streamlit page with title and icon
st.set_page_config(page_title="RAG Demo", page_icon="🤖", layout="wide")

# Initialize the session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_chunks" not in st.session_state:
    st.session_state.processed_chunks = []

# Sidebar for Knowledge Base Management
with st.sidebar:
    st.title("📚 Knowledge Base")
    st.subheader("Upload files to process into chunks")
    
    # Configuration for chunking
    with st.expander("Settings"):
        chunk_size = st.slider("Chunk Size", 100, 2000, 700)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100)
    
    uploaded_files = st.file_uploader(
        "Upload PDF or Markdown", 
        type=["pdf", "md"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process & Index Documents") and uploaded_files:
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        new_chunks = []
        
        progress_bar = st.progress(0)
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                if uploaded_file.name.endswith(".pdf"):
                    docs = chunker.load_pdf(uploaded_file, filename=uploaded_file.name)
                    chunks = chunker.chunk_documents(docs)
                elif uploaded_file.name.endswith(".md"):
                    docs = chunker.load_markdown(uploaded_file, filename=uploaded_file.name)
                    chunks = chunker.chunk_markdown_with_headers(docs[0]["content"], docs[0]["metadata"])
                
                new_chunks.extend(chunks)
                st.success(f"Processed {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        st.session_state.processed_chunks = new_chunks
        st.success(f"Indexing complete! Total chunks: {len(new_chunks)}")

    if st.session_state.processed_chunks:
        st.info(f"Total chunks in memory: {len(st.session_state.processed_chunks)}")
        if st.checkbox("Show Chunk Preview"):
            for i, c in enumerate(st.session_state.processed_chunks[:3]):
                st.caption(f"Chunk {i} ({c['metadata']['source']})")
                st.text(c['content'][:100] + "...")

# Display the main title and description of the application
st.title("🤖 RAG Question Answering")
st.markdown("""
This is an interactive RAG interface. Upload documents in the sidebar to process them into chunks, then ask questions about their content.
""")

# Display all previous messages from the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            with st.expander("Sources"):
                for source in message["citations"]:
                    st.write(f"- {source}")

# Handle user input and process the question
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Use processed chunks for a more "realistic" mock citation if they exist
        if st.session_state.processed_chunks:
            source_names = list(set([c['metadata']['source'] for c in st.session_state.processed_chunks[:2]]))
            dummy_answer = f"I've analyzed your {len(st.session_state.processed_chunks)} document chunks. To answer '{prompt}', I searched through {', '.join(source_names)}. This is a mock response demonstrating the RAG flow."
            dummy_citations = [f"Source: {name}" for name in source_names]
        else:
            dummy_answer = f"I don't have any uploaded documents yet! Please upload and process files in the sidebar. (Mock answer to: '{prompt}')"
            dummy_citations = ["No documents uploaded"]

        for chunk in dummy_answer.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        with st.expander("Sources"):
            for source in dummy_citations:
                st.write(f"- {source}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "citations": dummy_citations
        })
