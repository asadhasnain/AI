from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os

app = Flask(__name__)
CORS(app)

# Global variables for RAG components
chroma_client = None
collection = None
model = None

def initialize_rag():
    """Initialize the RAG system components"""
    global chroma_client, collection, model
    
    try:
        # Initialize Chroma persistent client
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or load a collection
        collection = chroma_client.get_or_create_collection(name="jina_docs")
        
        # Load Jina embedding model
        model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
        model.max_seq_length = 1024
        
        # Check if collection is empty and populate with sample data
        if collection.count() == 0:
            populate_sample_data()
            
        return True
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        return False

def populate_sample_data():
    """Populate the collection with sample documents"""
    texts = [
        "After Obama, Joe Biden was the president of United States.",
        "After Joe Biden, Donald Trump is the President of the United States.",
        "Till today July 15, 2025 Donald Trump is still the president.",
        "Barack Obama served as the 44th President of the United States from 2009 to 2017.",
        "Joe Biden served as the 46th President of the United States from 2021 to 2025.",
        "Donald Trump served as the 45th President from 2017 to 2021, and became the 47th President in 2025."
    ]
    
    # Get embeddings and convert to list of lists
    embeddings = model.encode(texts).tolist()
    
    # Auto-generate unique ids
    ids = [f"doc_{i}" for i in range(len(texts))]
    
    # Add to Chroma
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids
    )

def ask_mistral(prompt: str) -> str:
    """Send a prompt to the Mistral model via Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"Error: Unable to connect to Mistral model (Status: {response.status_code})"
    except requests.exceptions.RequestException as e:
        return f"Error: Unable to connect to Ollama server. Make sure Ollama is running and Mistral model is available."

def query_rag(question: str) -> dict:
    """Query the RAG system and return the response"""
    try:
        # Get query embedding
        query_embedding = model.encode([question]).tolist()
        
        # Search for similar documents
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        
        retrieved_docs_lists = results["documents"]
        all_docs = sum(retrieved_docs_lists, [])
        
        # Prepare context for the prompt
        context = "\n".join(all_docs)
        
        prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        # Get LLM response
        answer = ask_mistral(prompt)
        
        return {
            "success": True,
            "answer": answer,
            "context": all_docs,
            "question": question
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": "Sorry, I encountered an error while processing your question."
        }

@app.route('/')
def index():
    """Serve the chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = data['message'].strip()
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    # Query the RAG system
    result = query_rag(user_message)
    
    return jsonify(result)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "rag_initialized": chroma_client is not None and collection is not None,
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    print("Initializing RAG system...")
    if initialize_rag():
        print("RAG system initialized successfully!")
        print("Starting Flask server...")
        print("Open http://localhost:5000 in your browser to access the chat interface")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize RAG system. Please check your setup.")
