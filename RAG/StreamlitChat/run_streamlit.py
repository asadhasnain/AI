#!/usr/bin/env python3
"""
Streamlit Chat Launcher for AI RAG System

This script starts the Streamlit application for the RAG chat interface.

Usage:
    python run_streamlit.py
    
Make sure Ollama is running before starting:
    ollama serve
    ollama pull mistral
"""

import sys
import subprocess
import requests
import os

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
    print("🚀 Starting AI RAG Streamlit Chat Interface...")
    print("=" * 55)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit is not installed!")
        print("Please install dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Check Ollama server (optional - app will show status)
    print("🔍 Checking Ollama server...")
    if check_ollama_server():
        print("✅ Ollama server is running")
        
        # Check Mistral model
        if check_mistral_model():
            print("✅ Mistral model is available")
        else:
            print("⚠️  Mistral model not found - you can still use the app but won't be able to chat")
            print("   To fix: ollama pull mistral")
    else:
        print("⚠️  Ollama server not running - you can still upload documents")
        print("   To fix: ollama serve")
    
    print("\n🌐 Starting Streamlit application...")
    print("📱 The app will open in your browser automatically")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 55)
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
