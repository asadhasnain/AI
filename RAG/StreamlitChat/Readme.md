# AI RAG Streamlit Chat Interface

A modern, interactive Streamlit-based chat interface for Retrieval-Augmented Generation (RAG) that allows users to upload documents and chat with them in real-time.

## 🌟 Features

### Document Processing
- **Multiple File Formats**: Support for PDF, DOCX, and TXT files
- **Bulk Upload**: Upload multiple documents at once
- **Manual Text Input**: Add text directly through the interface
- **Smart Chunking**: Automatically splits large documents into manageable chunks
- **Real-time Processing**: Documents are processed and embedded instantly

### Chat Interface
- **Interactive Chat**: Real-time conversation with your documents
- **Source Citations**: See exactly which documents were used for each answer
- **Context Display**: Expandable view of source document excerpts
- **Chat History**: Persistent conversation history during the session
- **Error Handling**: Clear feedback when issues occur

### System Management
- **Live Status Monitoring**: Real-time checks for Ollama server and model availability
- **Collection Statistics**: View document count and collection info
- **Clear Functions**: Reset chat history or clear all documents
- **Settings Panel**: Adjust retrieval parameters

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed and running
- Mistral model pulled

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama server:**
   ```bash
   ollama serve
   ```

3. **Pull Mistral model:**
   ```bash
   ollama pull mistral
   ```

4. **Run the Streamlit app:**
   ```bash
   python run_streamlit.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and go to <http://localhost:8501>

## 📖 How to Use

### 1. Upload Documents
- Use the sidebar to upload PDF, DOCX, or TXT files
- Click "Process Documents" to add them to the collection
- Or manually add text using the text area

### 2. Start Chatting
- Type questions about your documents in the chat input
- The AI will search through your documents and provide answers
- Click "Source Documents" to see which documents were used

### 3. Manage Your Collection
- View collection statistics in the sidebar
- Clear individual chats or the entire document collection
- Adjust settings like the number of results to retrieve

## 🏗️ Architecture

```text
StreamlitChat/
├── app.py              # Main Streamlit application
├── run_streamlit.py    # Launcher script with system checks
├── requirements.txt    # Python dependencies
├── Readme.md          # This file
└── chroma_db/         # Vector database (auto-created)
```

## 🔧 Configuration

The app uses sensible defaults but can be customized:

- **Embedding Model**: `jinaai/jina-embeddings-v2-base-en`
- **Chunk Size**: 1000 characters with 200 character overlap
- **Default Results**: 3 most relevant documents
- **Ollama URL**: `http://localhost:11434`
- **Model**: `mistral`

## 📡 Core Components

### DocumentProcessor
Handles file upload and text extraction:
- PDF text extraction using PyPDF2
- DOCX processing with python-docx
- TXT file handling with proper encoding

### RAGSystem
Core RAG functionality:
- ChromaDB for vector storage
- Sentence transformers for embeddings
- Ollama integration for LLM responses
- Document chunking and retrieval

### Streamlit UI
Modern, responsive interface:
- Real-time status monitoring
- File upload with drag-and-drop
- Interactive chat with message history
- Expandable source document view

## 🎨 User Interface

### Sidebar Features
- **System Status**: Live monitoring of Ollama and model availability
- **Document Upload**: Drag-and-drop file upload interface
- **Manual Text Entry**: Direct text input option
- **Collection Management**: Statistics and clear functions

### Main Chat Area
- **Message History**: Persistent chat history with role-based styling
- **Source Citations**: Expandable sections showing source documents
- **Real-time Responses**: Streaming-style message display
- **Error Feedback**: Clear error messages and suggestions

### Settings Panel
- **Retrieval Settings**: Adjust number of results
- **Chat Management**: Clear history and reset functions
- **Usage Instructions**: Built-in help and guidance

## 🔍 Advanced Features

### Smart Document Chunking
- Automatic text splitting with sentence boundary detection
- Configurable chunk size and overlap
- Preserves context across chunk boundaries

### Real-time System Monitoring
- Ollama server connectivity checks
- Model availability validation
- Collection status monitoring

### Error Recovery
- Graceful handling of connection issues
- Clear error messages with actionable suggestions
- System status indicators

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama Not Connected**
   - Start Ollama server: `ollama serve`
   - Check if running on port 11434

2. **Mistral Model Not Found**
   - Pull the model: `ollama pull mistral`
   - Verify with: `ollama list`

3. **File Upload Issues**
   - Check file format (PDF, DOCX, TXT only)
   - Ensure files are not corrupted
   - Try smaller files if processing fails

4. **Slow Performance**
   - Reduce chunk size for faster processing
   - Limit number of retrieved results
   - Use smaller documents for testing

### System Requirements
- **RAM**: At least 4GB (8GB+ recommended for large documents)
- **Storage**: Space for ChromaDB and uploaded documents
- **Network**: Internet connection for initial model downloads

## 🚀 Future Enhancements

- [ ] Support for more file formats (Excel, PowerPoint, etc.)
- [ ] Advanced document preprocessing options
- [ ] User authentication and document privacy
- [ ] Export chat conversations
- [ ] Custom embedding models
- [ ] Document versioning and updates
- [ ] Batch document processing
- [ ] Integration with cloud storage services

## 💡 Tips for Best Results

1. **Document Quality**: Use well-formatted, clear documents
2. **Chunk Size**: Adjust based on your document types
3. **Question Style**: Ask specific, clear questions
4. **Context**: Provide context in your questions when needed
5. **Source Review**: Always check the source documents for accuracy

## 🤝 Contributing

This is part of the AI RAG samples repository. Feel free to suggest improvements or report issues!
