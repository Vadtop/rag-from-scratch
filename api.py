from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests
import numpy as np
import os
from dotenv import load_dotenv
from vector_store import VectorStore
from google_sheets import log_query

load_dotenv()  # читаем OPENROUTER_API_KEY из .env

app = FastAPI(title="RAG API", version="3.0")

# ========== НАСТРОЙКИ ==========
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "deepseek/deepseek-chat"  # дешевле, через OpenRouter

# Память диалога для /agent endpoint (in-memory, per session)
_agent_sessions: dict[str, list] = {}


# Глобальное хранилище chunks (в памяти)
vector_store = VectorStore()

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
            "model": LLM_MODEL,
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
        "version": "2.1",
        "endpoints": {
            "POST /upload": "Upload document (Manual RAG)",
            "POST /query": "Ask question (Manual RAG)",
            "POST /upload_langchain": "Upload document (LangChain RAG)",
            "POST /query_langchain": "Ask question (LangChain RAG)",
            "GET /stats": "Get statistics",
            "DELETE /reset": "Clear database"
        },
        "approaches": {
            "manual": "Full control, custom logic (v2.0)",
            "langchain": "Fast development, framework benefits (v2.1)"
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
    
    # Создаём embeddings и сохраняем в ChromaDB
    for i, chunk in enumerate(text_chunks):
        embedding = get_embedding(chunk)
        
        chunk_id = f"{file.filename}_chunk_{i}"
        
        vector_store.add_chunk(
            chunk_id=chunk_id,
            content=chunk,
            embedding=embedding,
            metadata={
                "source": file.filename,
                "chunk_id": i,
                "total_chunks": len(text_chunks)
            }
        )
    
    return {
        "status": "success",
        "filename": file.filename,
        "chunks_created": len(text_chunks),
        "total_chunks_in_db": vector_store.count()
    }

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Отвечает на вопрос используя RAG"""
    
    if vector_store.count() == 0:
        return QueryResponse(
            answer="No documents uploaded yet. Please upload documents first.",
            sources=[],
            chunks_used=[]
        )
    
    # Embedding запроса
    query_emb = get_embedding(req.query)
    
    # Поиск через ChromaDB
    results = vector_store.search(query_emb, top_k=req.top_k)
    
    # Извлекаем данные из результатов
    documents = results['documents'][0]  # list of texts
    metadatas = results['metadatas'][0]  # list of metadata dicts
    distances = results['distances'][0]  # list of distances (lower = better)
    
    # Формируем контекст
    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        context_parts.append(
            f"[{meta['source']}, chunk {meta['chunk_id']+1}/{meta['total_chunks']}]\n{doc}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Генерация
    prompt = f"""Answer the question based on this context.

Context:
{context}

Question: {req.query}

Answer (be concise):"""
    
    answer = get_completion([{"role": "user", "content": prompt}])
    
    sources = list(set([meta['source'] for meta in metadatas]))
    chunks_used = [
        {
            "source": meta['source'],
            "chunk_id": meta['chunk_id'] + 1,
            "distance": float(f"{dist:.3f}")
        }
        for meta, dist in zip(metadatas, distances)
    ]
    
    # Логируем в Google Sheets
    log_query(
        question=req.query,
        answer=answer,
        sources=sources,
        model=LLM_MODEL,
        chunks_used=len(chunks_used),
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        chunks_used=chunks_used
    )


# ========== AI AGENT ENDPOINT ==========

class AgentRequest(BaseModel):
    query: str
    session_id: str = "default"  # для multi-turn диалога


class AgentResponse(BaseModel):
    answer: str
    sources: list
    session_id: str
    turn: int


@app.post("/agent", response_model=AgentResponse)
def agent_query(req: AgentRequest):
    """
    AI-агент с памятью диалога.
    Использует RAG для поиска контекста + помнит историю разговора.
    Логирует каждый вопрос/ответ в Google Sheets.
    """
    # Инициализация сессии
    if req.session_id not in _agent_sessions:
        _agent_sessions[req.session_id] = []

    history = _agent_sessions[req.session_id]

    # RAG: найти релевантный контекст
    context = ""
    sources = []
    if vector_store.count() > 0:
        query_emb = get_embedding(req.query)
        results = vector_store.search(query_emb, top_k=3)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        context_parts = [
            f"[{m['source']}] {d}" for d, m in zip(documents, metadatas)
        ]
        context = "\n\n".join(context_parts)
        sources = list(set(m["source"] for m in metadatas))

    # Системный промпт
    system_prompt = (
        "Ты AI-агент с доступом к базе знаний. "
        "Отвечай на вопросы используя предоставленный контекст. "
        "Если контекста нет — отвечай из общих знаний, но предупреди об этом. "
        "Помни историю диалога."
    )
    if context:
        system_prompt += f"\n\nБаза знаний:\n{context}"

    # Сборка messages: system + история + новый вопрос
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-10:])  # последние 10 сообщений
    messages.append({"role": "user", "content": req.query})

    # LLM ответ
    answer = get_completion(messages)

    # Сохранить в историю
    history.append({"role": "user", "content": req.query})
    history.append({"role": "assistant", "content": answer})

    # Логировать в Google Sheets
    log_query(
        question=req.query,
        answer=answer,
        sources=sources,
        model=LLM_MODEL,
        chunks_used=len(sources),
    )

    return AgentResponse(
        answer=answer,
        sources=sources,
        session_id=req.session_id,
        turn=len(history) // 2,
    )


@app.delete("/agent/{session_id}")
def clear_session(session_id: str):
    """Очистить историю диалога сессии."""
    _agent_sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/stats")
def get_stats():
    """Статистика базы знаний"""
    sources = vector_store.get_all_sources()
    
    return {
        "total_chunks": vector_store.count(),
        "total_documents": len(sources),
        "documents": sources
    }


@app.delete("/reset")
def reset():
    """Очищает базу знаний"""
    vector_store.clear()
    return {"status": "success", "message": "Database cleared"}


# ========== LANGCHAIN RAG ENDPOINT ==========

_langchain_rag = None

def _get_langchain_rag():
    global _langchain_rag
    if _langchain_rag is None:
        from step2_langchain import LangChainRAG
        _langchain_rag = LangChainRAG()
    return _langchain_rag

@app.post("/query_langchain")
def query_langchain(req: QueryRequest):
    """Отвечает на вопрос используя LangChain RAG"""
    result = _get_langchain_rag().query(req.query)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "method": "langchain",
        "chunks_used": result.get("chunks_used", 0)
    }


# ========== ЗАПУСК ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
