#!/usr/bin/env python3
"""
Chat UI Runner for AI RAG System

This script starts the Flask web application that provides a chat interface
for the RAG (Retrieval-Augmented Generation) system.

Before running this script, make sure:
1. All dependencies are installed: pip install -r requirements.txt
2. Ollama server is running: ollama serve
3. Mistral model is available: ollama pull mistral

Usage:
    python run_chat.py
    
Then open http://localhost:5000 in your browser.
"""

import sys
import subprocess
import requests
import time

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def check_mistral_model():
    """Check if Mistral model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any("mistral" in model.get("name", "").lower() for model in models)
    except requests.exceptions.RequestException:
        pass
    return False

def main():
    print("🚀 Starting AI RAG Chat Interface...")
    print("=" * 50)
    
    # Check Ollama server
    print("🔍 Checking Ollama server...")
    if not check_ollama_server():
        print("❌ Ollama server is not running!")
        print("Please start Ollama server first:")
        print("   ollama serve")
        sys.exit(1)
    print("✅ Ollama server is running")
    
    # Check Mistral model
    print("🔍 Checking Mistral model...")
    if not check_mistral_model():
        print("❌ Mistral model is not available!")
        print("Please pull the Mistral model first:")
        print("   ollama pull mistral")
        sys.exit(1)
    print("✅ Mistral model is available")
    
    print("🌐 Starting Flask web server...")
    print("📱 Open http://localhost:5000 in your browser to access the chat interface")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run the Flask app
    try:
        from app import app, initialize_rag
        
        # Initialize RAG system
        if not initialize_rag():
            print("❌ Failed to initialize RAG system")
            sys.exit(1)
            
        # Start the Flask server
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        print("Please make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Chat interface stopped")
    except Exception as e:
        print(f"❌ Error starting chat interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
