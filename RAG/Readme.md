# AI RAG System

A simple Retrieval-Augmented Generation system using Jina embeddings and Mistral LLM all running locally.

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

Run the main script:

```bash
python JinaaiEmbeddings.py
```

The system will answer questions about US presidents using the embedded documents.

## How it works

1. Documents are embedded using Jina AI
2. User query is embedded and matched against documents
3. Relevant documents are sent to Mistral for answer generation