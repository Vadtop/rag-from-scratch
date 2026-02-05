from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()  # читаем OPENROUTER_API_KEY из .env

app = FastAPI(title="RAG API", version="1.0")

# ========== НАСТРОЙКИ ==========
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"


# Глобальное хранилище chunks (в памяти)
chunks = []

# ========== API ФУНКЦИИ ==========
def get_embedding(text):
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/text-embedding-3-small",
            "input": text
        }
    )
    return response.json()["data"][0]["embedding"]

def get_completion(messages):
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# ========== CHUNKING ==========
def chunk_text(text, chunk_size=500, overlap=100):
    chunks_list = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks_list.append(chunk)
        start += (chunk_size - overlap)
    return chunks_list

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# ========== MODELS ==========
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list
    chunks_used: list

# ========== ENDPOINTS ==========
@app.get("/")
def root():
    return {
        "message": "RAG API",
        "version": "1.0",
        "endpoints": {
            "POST /upload": "Upload document",
            "POST /query": "Ask question",
            "GET /stats": "Get statistics"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Загружает документ и создаёт embeddings"""
    
    # Читаем файл
    content = await file.read()
    text = content.decode("utf-8")
    
    # Chunking
    text_chunks = chunk_text(text, chunk_size=500, overlap=100)
    
    # Создаём embeddings
    for i, chunk in enumerate(text_chunks):
        embedding = get_embedding(chunk)
        
        chunks.append({
            "content": chunk,
            "source": file.filename,
            "chunk_id": i,
            "total_chunks": len(text_chunks),
            "embedding": embedding
        })
    
    return {
        "status": "success",
        "filename": file.filename,
        "chunks_created": len(text_chunks),
        "total_chunks_in_db": len(chunks)
    }

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Отвечает на вопрос используя RAG"""
    
    if not chunks:
        return QueryResponse(
            answer="No documents uploaded yet. Please upload documents first.",
            sources=[],
            chunks_used=[]
        )
    
    # Embedding запроса
    query_emb = get_embedding(req.query)
    
    # Поиск
    results = []
    for chunk in chunks:
        sim = cosine_similarity(query_emb, chunk["embedding"])
        results.append((chunk, sim))
    
    results.sort(key=lambda x: x[1], reverse=True)
    top_chunks = results[:req.top_k]
    
    # Контекст
    context_parts = []
    for chunk, score in top_chunks:
        context_parts.append(
            f"[{chunk['source']}, chunk {chunk['chunk_id']+1}/{chunk['total_chunks']}]\n{chunk['content']}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Генерация
    prompt = f"""Answer the question based on this context.

Context:
{context}

Question: {req.query}

Answer (be concise):"""
    
    answer = get_completion([{"role": "user", "content": prompt}])
    
    sources = list(set([chunk['source'] for chunk, _ in top_chunks]))
    chunks_used = [
        {
            "source": chunk['source'],
            "chunk_id": chunk['chunk_id'] + 1,
            "similarity": float(f"{score:.3f}")
        }
        for chunk, score in top_chunks
    ]
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        chunks_used=chunks_used
    )

@app.get("/stats")
def get_stats():
    """Статистика базы знаний"""
    sources = list(set([chunk['source'] for chunk in chunks]))
    
    return {
        "total_chunks": len(chunks),
        "total_documents": len(sources),
        "documents": sources
    }

@app.delete("/reset")
def reset():
    """Очищает базу знаний"""
    global chunks
    chunks = []
    return {"status": "success", "message": "Database cleared"}

# ========== ЗАПУСК ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
