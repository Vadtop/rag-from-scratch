from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import numpy as np
import os
from dotenv import load_dotenv
from vector_store import VectorStore
from google_sheets import log_query
from huggingface_rag import (
    embed_query,
    embed_texts,
    generate as hf_generate,
    generate_structured,
)

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
            "Content-Type": "application/json",
        },
        json={"model": "openai/text-embedding-3-small", "input": text},
    )
    return response.json()["data"][0]["embedding"]


def get_completion(messages):
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": LLM_MODEL, "messages": messages, "temperature": 0},
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
        start += chunk_size - overlap
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
        "version": "3.0",
        "endpoints": {
            "POST /upload": "Upload document to knowledge base",
            "POST /query": "Ask question (RAG + Google Sheets logging)",
            "POST /agent": "Multi-turn dialog agent with memory",
            "DELETE /agent/{session_id}": "Clear dialog session",
            "POST /query_langchain": "Ask question via LangChain RAG",
            "POST /query_hf": "RAG with local HuggingFace model (no API needed)",
            "POST /structured": "Structured JSON output from schema + local LLM",
            "GET /embeddings/info": "Embedding model info",
            "POST /embeddings": "Compute embeddings (local sentence-transformers)",
            "GET /stats": "Get knowledge base statistics",
            "DELETE /reset": "Clear knowledge base",
        },
        "model": "deepseek/deepseek-chat via OpenRouter",
    }


@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    return """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 16px 24px; background: #1a1a1a; border-bottom: 1px solid #2a2a2a; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 600; }
  header span { font-size: 12px; color: #666; background: #2a2a2a; padding: 3px 8px; border-radius: 12px; }
  #messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
  .msg { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 14px; }
  .msg.user { align-self: flex-end; background: #2563eb; color: white; border-bottom-right-radius: 4px; }
  .msg.bot { align-self: flex-start; background: #1e1e1e; border: 1px solid #2a2a2a; border-bottom-left-radius: 4px; }
  .msg.bot .sources { margin-top: 8px; font-size: 12px; color: #666; }
  .typing { color: #666; font-style: italic; font-size: 13px; }
  #input-area { padding: 16px 24px; background: #1a1a1a; border-top: 1px solid #2a2a2a; display: flex; gap: 10px; }
  #input { flex: 1; background: #2a2a2a; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; color: #e0e0e0; font-size: 14px; outline: none; }
  #input:focus { border-color: #2563eb; }
  button { background: #2563eb; color: white; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-size: 14px; font-weight: 500; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #333; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>RAG Agent</h1>
  <span>DeepSeek + ChromaDB</span>
</header>
<div id="messages">
  <div class="msg bot">Привет! Я AI-агент с базой знаний. Спроси меня про RAG, AI-агентов, Python или машинное обучение.</div>
</div>
<div id="input-area">
  <input id="input" type="text" placeholder="Задай вопрос..." onkeydown="if(event.key==='Enter') send()">
  <button id="btn" onclick="send()">Отправить</button>
</div>
<script>
const SESSION_ID = 'session_' + Math.random().toString(36).substr(2, 9);
async function send() {
  const input = document.getElementById('input');
  const btn = document.getElementById('btn');
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  btn.disabled = true;
  addMsg(query, 'user');
  const typing = addMsg('Думаю...', 'bot typing');
  try {
    const res = await fetch('/agent', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, session_id: SESSION_ID})
    });
    const data = await res.json();
    typing.remove();
    const sources = data.sources && data.sources.length ? '<div class="sources">Источники: ' + data.sources.join(', ') + '</div>' : '';
    addMsgHtml(data.answer.replace(/\\n/g, '<br>') + sources, 'bot');
  } catch(e) {
    typing.remove();
    addMsg('Ошибка соединения', 'bot');
  }
  btn.disabled = false;
  input.focus();
}
function addMsg(text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  document.getElementById('messages').appendChild(div);
  div.scrollIntoView();
  return div;
}
function addMsgHtml(html, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = html;
  document.getElementById('messages').appendChild(div);
  div.scrollIntoView();
  return div;
}
</script>
</body>
</html>
"""


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
                "total_chunks": len(text_chunks),
            },
        )

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_created": len(text_chunks),
        "total_chunks_in_db": vector_store.count(),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Отвечает на вопрос используя RAG"""

    if vector_store.count() == 0:
        return QueryResponse(
            answer="No documents uploaded yet. Please upload documents first.",
            sources=[],
            chunks_used=[],
        )

    # Embedding запроса
    query_emb = get_embedding(req.query)

    # Поиск через ChromaDB
    results = vector_store.search(query_emb, top_k=req.top_k)

    # Извлекаем данные из результатов
    documents = results["documents"][0]  # list of texts
    metadatas = results["metadatas"][0]  # list of metadata dicts
    distances = results["distances"][0]  # list of distances (lower = better)

    # Формируем контекст
    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        context_parts.append(
            f"[{meta['source']}, chunk {meta['chunk_id'] + 1}/{meta['total_chunks']}]\n{doc}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Генерация
    prompt = f"""Answer the question based on this context.

Context:
{context}

Question: {req.query}

Answer (be concise):"""

    answer = get_completion([{"role": "user", "content": prompt}])

    sources = list(set([meta["source"] for meta in metadatas]))
    chunks_used = [
        {
            "source": meta["source"],
            "chunk_id": meta["chunk_id"] + 1,
            "distance": float(f"{dist:.3f}"),
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

    return QueryResponse(answer=answer, sources=sources, chunks_used=chunks_used)


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
        context_parts = [f"[{m['source']}] {d}" for d, m in zip(documents, metadatas)]
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
        "documents": sources,
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
        "chunks_used": result.get("chunks_used", 0),
    }


# ========== HUGGINGFACE RAG ENDPOINTS ==========


class HFQueryRequest(BaseModel):
    query: str
    top_k: int = 3


class HFQueryResponse(BaseModel):
    answer: str
    sources: list
    chunks_used: int
    model: str = "Qwen2.5-1.5B-Instruct (local HuggingFace)"


class StructuredRequest(BaseModel):
    prompt: str
    schema: dict


class StructuredResponse(BaseModel):
    result: dict
    model: str = "Qwen2.5-1.5B-Instruct (local HuggingFace)"


@app.post("/query_hf", response_model=HFQueryResponse)
def query_hf(req: HFQueryRequest):
    """RAG with local HuggingFace model + sentence-transformers embeddings (no API needed)."""
    if vector_store.count() == 0:
        return HFQueryResponse(
            answer="No documents uploaded yet.", sources=[], chunks_used=0
        )

    query_emb = embed_query(req.query)
    results = vector_store.search(query_emb, top_k=req.top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    for doc, meta in zip(documents, metadatas):
        context_parts.append(
            f"[{meta['source']}, chunk {meta['chunk_id'] + 1}/{meta['total_chunks']}]\n{doc}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"Answer the question based on this context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.query}\n\n"
        f"Answer (be concise):"
    )

    answer = hf_generate(prompt, max_new_tokens=256, temperature=0.3)
    sources = list(set(meta["source"] for meta in metadatas))

    return HFQueryResponse(answer=answer, sources=sources, chunks_used=len(documents))


@app.post("/structured", response_model=StructuredResponse)
def structured_output(req: StructuredRequest):
    """Generate structured JSON output from a Pydantic-style schema using local HuggingFace model."""
    result = generate_structured(req.prompt, req.schema, max_new_tokens=512)
    return StructuredResponse(result=result)


@app.get("/embeddings/info")
def embeddings_info():
    """Info about the local embedding model."""
    from huggingface_rag import get_embedding_model

    model = get_embedding_model()
    return {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "provider": "HuggingFace (local, no API)",
    }


@app.post("/embeddings")
def compute_embeddings(texts: list[str]):
    """Compute embeddings for a list of texts using local sentence-transformers."""
    embs = embed_texts(texts)
    return {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": len(embs[0]),
        "count": len(embs),
        "embeddings": embs,
    }


# ========== ЗАПУСК ==========
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
