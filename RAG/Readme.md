# AI RAG System

A Retrieval-Augmented Generation system using Jina embeddings and Mistral LLM with both command-line and web chat interfaces.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install Ollama from [https://ollama.com/download](https://ollama.com/download)

3. Start Ollama and pull Mistral:

   ```bash
   ollama serve
   ```
   
   In another terminal:

   ```bash
   ollama pull mistral
   ```

## Usage

### Option 1: Web Chat Interface (Recommended)

Run the chat interface:

```bash
python run_chat.py
```

Then open <http://localhost:5000> in your browser for an interactive chat experience.

### Option 2: Command Line

Run the original script:

```bash
python JinaaiEmbeddings.py
```

## Features

- **Web Chat Interface**: Modern, responsive chat UI
- **Real-time Responses**: Interactive conversation with the AI
- **Context Display**: Shows which documents were used for answers
- **Error Handling**: Graceful handling of connection issues
- **Health Monitoring**: Automatic system status checks

## Project Structure

```text
RAG/
├── app.py                 # Flask web application
├── run_chat.py           # Chat interface launcher
├── JinaaiEmbeddings.py   # Original command-line script
├── templates/
│   └── index.html        # Chat interface HTML
├── requirements.txt      # Dependencies
└── chroma_db/           # Vector database (auto-created)
```

## How it works

1. Documents are embedded using Jina AI
2. User query is embedded and matched against documents
3. Relevant documents are sent to Mistral for answer generation
4. Results are displayed in a beautiful web interface