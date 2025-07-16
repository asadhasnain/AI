# AI Samples Repository

This repository contains AI-related projects and demonstrations showcasing various artificial intelligence techniques and frameworks.

## Current Samples

### 1. Retrieval-Augmented Generation (RAG)

**Location:** `RAG/`

A complete RAG system with multiple interface options for intelligent question answering using semantic search and large language models.

**Available Interfaces:**

#### Streamlit Chat App (Recommended)
- **Location:** `RAG/StreamlitChat/`
- **Features:**
  - Upload and chat with your own documents (PDF, DOCX, TXT)
  - Real-time document processing and embedding
  - Interactive chat interface with source citations
  - Live system monitoring and status checks
  - Document collection management

**Quick Start:**
```bash
cd RAG/StreamlitChat/
pip install -r requirements.txt
python run_streamlit.py
# Open http://localhost:8501 in your browser
```

#### Flask Web Chat
- **Location:** `RAG/ChatUI/`
- **Features:**
  - Web-based chat interface
  - RESTful API endpoints
  - Predefined knowledge base about US Presidents

**Quick Start:**
```bash
cd RAG/ChatUI/
pip install -r requirements.txt
python run_chat.py
# Open http://localhost:5000 in your browser
```

#### Console Application
- **Location:** `RAG/Console/`
- **Features:**
  - Simple command-line interface
  - Basic RAG functionality demonstration

**Quick Start:**
```bash
cd RAG/Console/
pip install -r requirements.txt
python JinaaiEmbeddings.py
```

**Core Technologies:**

- Jina AI embeddings for document vectorization
- ChromaDB for vector storage and similarity search
- Local Mistral LLM integration via Ollama
- Modern web interfaces with real-time capabilities

## Getting Started

Each sample project contains its own README with detailed setup and usage instructions. Navigate to the specific project directory for more information.

## Requirements

- Python 3.8+
- Internet connection for downloading models
- Sufficient RAM for running AI models locally

## Contributing

Feel free to add new AI samples and experiments to this repository. Each project should include:

- Clear documentation in a README file
- Requirements file for dependencies
- Example usage and setup instructions

