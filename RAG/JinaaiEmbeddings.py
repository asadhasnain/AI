import chromadb
from sentence_transformers import SentenceTransformer
import requests

# function to ask Mistral model
# Make sure you have the Mistral model running locally on port 11434
# ollama run mistral

def ask_mistral(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    return response.json().get("response", "").strip()


# Initialize Chroma persistent client correctly
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or load a collection
collection = chroma_client.get_or_create_collection(name="jina_docs")

# Load Jina embedding model
model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
texts = [
    "After Obama, Jeo Biden was the president of United State ?",
    "After Jeo Biden, Donald Trump is the President of the United States.",
    "Till today July 03, 2025 Donald Trump is still the president.",
]

model.max_seq_length = 1024

# Get embeddings and convert to list of lists
embeddings = model.encode(texts).tolist()

# Auto-generate unique ids like: doc_0, doc_1, ...
ids = [f"doc_{i}" for i in range(len(texts))]

# print(f"\nEmbeddings:{embeddings}\n")

# Add to Chroma
collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids
)

# Query the collection and find the most similar documents
query = "List the last three presidents of United States?"
query_embedding = model.encode([query]).tolist()

results = collection.query(query_embeddings=query_embedding, n_results=2)

retrieved_docs_lists = results["documents"]  # list of lists
all_docs = sum(retrieved_docs_lists, [])     # flatten to single list

# Prepare context for the prompt
context = "\n".join(all_docs)

prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

# Get LLM response
answer = ask_mistral(prompt)

print(f"\nPrompt: {query}\n")

print(f"\nAnswer: {answer}\n")
