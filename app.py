import os
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb

USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "1") == "1"  # Set default to mock mode

if not USE_MOCK_LLM:
    import ollama

app = FastAPI()

# Initialize ChromaDB
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

class QueryRequest(BaseModel):
    q: str

@app.post("/query")
def query(request: QueryRequest):
    q = request.q.strip()
    
    # Query the collection safely
    results = collection.query(query_texts=[q], n_results=1)
    documents = results.get("documents", [])
    context = documents[0][0] if documents and documents[0] else ""

    if USE_MOCK_LLM:
        # Return context directly in mock mode
        if not context:
            context = "Kubernetes is a container orchestration platform."
        return {"answer": context}

    # Production mode with Ollama
    answer = ollama.generate(
        model="tinyllama",
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
    )
    return {"answer": answer["response"]}
