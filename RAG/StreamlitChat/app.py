"""
AI RAG Chat Interface with Streamlit

A Streamlit-based chat interface for Retrieval-Augmented Generation (RAG)
that allows users to upload documents and chat with them in real-time.
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import time
import os
from typing import List, Dict, Any
import PyPDF2
import docx
import io

# Page configuration
st.set_page_config(
    page_title="AI RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DocumentProcessor:
    """Handle document processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file) -> str:
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return ""
    
    @staticmethod
    def process_uploaded_file(uploaded_file) -> str:
        """Process uploaded file and extract text"""
        if uploaded_file is None:
            return ""
        
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            return DocumentProcessor.extract_text_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return DocumentProcessor.extract_text_from_docx(uploaded_file)
        elif file_type == "text/plain":
            return DocumentProcessor.extract_text_from_txt(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return ""

class RAGSystem:
    """RAG System for document retrieval and answer generation"""
    
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.model = None
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "mistral"
        
    def initialize(self) -> bool:
        """Initialize the RAG system"""
        try:
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(name="streamlit_docs")
            
            # Load embedding model
            with st.spinner("Loading embedding model..."):
                self.model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
                self.model.max_seq_length = 1024
            
            return True
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_model_availability(self) -> bool:
        """Check if Mistral model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any("mistral" in model.get("name", "").lower() for model in models)
        except:
            pass
        return False
    
    def add_documents(self, documents: List[str]) -> bool:
        """Add documents to the collection"""
        if not documents:
            return False
        
        try:
            with st.spinner("Processing documents..."):
                # Filter out empty documents
                valid_docs = [doc.strip() for doc in documents if doc.strip()]
                
                if not valid_docs:
                    st.warning("No valid documents to add")
                    return False
                
                # Get embeddings
                embeddings = self.model.encode(valid_docs).tolist()
                
                # Generate unique IDs
                current_count = self.collection.count()
                ids = [f"doc_{current_count + i}_{int(time.time())}" for i in range(len(valid_docs))]
                
                # Add to collection
                self.collection.add(
                    documents=valid_docs,
                    embeddings=embeddings,
                    ids=ids
                )
                
                return True
        except Exception as e:
            st.error(f"Error adding documents: {e}")
            return False
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within the last 100 characters
                chunk_text = text[start:end]
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size - 200:  # If we found a good break point
                    end = start + break_point + 1
            
            chunks.append(text[start:end].strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """Query the RAG system"""
        if not question.strip():
            return {"success": False, "error": "Empty question"}
        
        try:
            # Check if collection has documents
            if self.collection.count() == 0:
                return {
                    "success": False,
                    "error": "No documents in the collection. Please upload some documents first."
                }
            
            # Get query embedding
            query_embedding = self.model.encode([question]).tolist()
            
            # Search for similar documents
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, self.collection.count())
            )
            
            # Extract documents
            retrieved_docs = results["documents"][0] if results["documents"] else []
            
            if not retrieved_docs:
                return {
                    "success": False,
                    "error": "No relevant documents found for your question."
                }
            
            # Prepare context
            context = "\n\n".join(retrieved_docs)
            
            # Create prompt
            prompt = f"""Answer the question using the context below. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}
Answer:"""
            
            # Get LLM response
            answer = self.ask_mistral(prompt)
            
            return {
                "success": True,
                "answer": answer,
                "context": retrieved_docs,
                "question": question
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing query: {str(e)}"
            }
    
    def ask_mistral(self, prompt: str) -> str:
        """Send prompt to Mistral model"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"Error: Unable to connect to Mistral model (Status: {response.status_code})"
                
        except Exception as e:
            return f"Error: Unable to connect to Ollama server. {str(e)}"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        return {
            "total_documents": self.collection.count(),
            "collection_name": "streamlit_docs"
        }
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            # Delete and recreate collection
            self.chroma_client.delete_collection("streamlit_docs")
            self.collection = self.chroma_client.get_or_create_collection(name="streamlit_docs")
            return True
        except Exception as e:
            st.error(f"Error clearing collection: {e}")
            return False

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🤖 AI RAG Chat Assistant")
    st.markdown("Upload documents and chat with your AI assistant!")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # System status
        st.subheader("🔧 System Status")
        
        if not st.session_state.system_initialized:
            with st.spinner("Initializing RAG system..."):
                if st.session_state.rag_system.initialize():
                    st.session_state.system_initialized = True
                    st.success("✅ RAG system initialized!")
                else:
                    st.error("❌ Failed to initialize RAG system")
                    st.stop()
        
        # Check Ollama connection
        ollama_status = st.session_state.rag_system.check_ollama_connection()
        model_status = st.session_state.rag_system.check_model_availability()
        
        if ollama_status:
            st.success("✅ Ollama server connected")
        else:
            st.error("❌ Ollama server not accessible")
            st.markdown("Please start Ollama server: `ollama serve`")
        
        if model_status:
            st.success("✅ Mistral model available")
        else:
            st.error("❌ Mistral model not found")
            st.markdown("Please pull Mistral model: `ollama pull mistral`")
        
        st.divider()
        
        # Document upload
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                documents = []
                
                for uploaded_file in uploaded_files:
                    st.write(f"Processing: {uploaded_file.name}")
                    text = DocumentProcessor.process_uploaded_file(uploaded_file)
                    
                    if text:
                        # Split text into chunks
                        chunks = st.session_state.rag_system.split_text_into_chunks(text)
                        documents.extend(chunks)
                        st.success(f"✅ {uploaded_file.name} processed ({len(chunks)} chunks)")
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}")
                
                if documents:
                    if st.session_state.rag_system.add_documents(documents):
                        st.success(f"🎉 Added {len(documents)} document chunks to collection!")
                    else:
                        st.error("Failed to add documents to collection")
        
        st.divider()
        
        # Manual text input
        st.subheader("✏️ Add Text Manually")
        manual_text = st.text_area(
            "Enter text",
            height=150,
            placeholder="Paste or type your text here..."
        )
        
        if st.button("Add Text"):
            if manual_text.strip():
                chunks = st.session_state.rag_system.split_text_into_chunks(manual_text)
                if st.session_state.rag_system.add_documents(chunks):
                    st.success(f"✅ Added {len(chunks)} text chunks!")
                else:
                    st.error("Failed to add text")
            else:
                st.warning("Please enter some text")
        
        st.divider()
        
        # Collection stats
        st.subheader("📊 Collection Stats")
        stats = st.session_state.rag_system.get_collection_stats()
        if "error" not in stats:
            st.metric("Total Documents", stats["total_documents"])
        
        # Clear collection
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.rag_system.clear_collection():
                st.success("Collection cleared!")
                st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show context for assistant messages
                    if message["role"] == "assistant" and "context" in message:
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(message["context"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(f"_{doc[:200]}..._" if len(doc) > 200 else f"_{doc}_")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_system.query(prompt)
                
                if result["success"]:
                    st.markdown(result["answer"])
                    
                    # Add to message history with context
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "context": result["context"]
                    })
                    
                    # Show context
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(result["context"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(f"_{doc[:200]}..._" if len(doc) > 200 else f"_{doc}_")
                else:
                    error_msg = result.get("error", "An unknown error occurred")
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}"
                    })
    
    with col2:
        st.subheader("⚙️ Settings")
        
        # Number of results
        n_results = st.slider("Results to retrieve", 1, 10, 3)
        
        # Clear chat history
        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Instructions
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
        2. **Add Text**: Manually add text through the text area
        3. **Ask Questions**: Type questions about your documents in the chat
        4. **View Sources**: Click on "Source Documents" to see which documents were used
        5. **Manage Collection**: View stats and clear documents as needed
        """)

if __name__ == "__main__":
    main()
