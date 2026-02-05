import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # читаем OPENROUTER_API_KEY из .env

API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"


# ========== API ФУНКЦИИ ==========
def get_embedding(text):
    """Получает embedding через requests"""
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
    """Получает ответ от LLM через requests"""
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
def chunk_text(text, chunk_size=150, overlap=50):
    """Разбивает текст на куски с перекрытием"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        print(f"Chunk {len(chunks)}: start={start}, end={end}, len={len(chunk)}")  # ← ДОБАВЬ
        
        if chunk.strip():
            chunks.append(chunk)
        
        start += (chunk_size - overlap)
    
    return chunks

# ========== ЗАГРУЗКА ==========
def load_documents_with_chunking(folder_path="documents"):
    """Загружает документы и разбивает на чанки"""
    chunks_data = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            chunks = chunk_text(content, chunk_size=500, overlap=100)
            
            for i, chunk in enumerate(chunks):
                chunks_data.append({
                    "content": chunk,
                    "source": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
            
            print(f"📄 {filename}: {len(chunks)} chunks")
    
    return chunks_data

# ========== COSINE SIMILARITY ==========
def cosine_similarity(vec1, vec2):
    """Вычисляет косинусное сходство"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# ========== MAIN ==========
print("📂 Загружаю документы с chunking...")
chunks = load_documents_with_chunking()
print(f"✅ Всего chunks: {len(chunks)}\n")

print("⏳ Создаю embeddings...")
for i, chunk in enumerate(chunks):
    chunk["embedding"] = get_embedding(chunk["content"])
    print(f"  {i+1}/{len(chunks)}", end="\r")
print("\n✅ Embeddings готовы!\n")

# ========== RAG ==========
def rag_query(query, top_k=3):
    """RAG pipeline"""
    
    # 1. Embedding запроса
    query_emb = get_embedding(query)
    
    # 2. Поиск
    results = []
    for chunk in chunks:
        sim = cosine_similarity(query_emb, chunk["embedding"])
        results.append((chunk, sim))
    
    results.sort(key=lambda x: x[1], reverse=True)
    top_chunks = results[:top_k]
    
    # 3. Контекст
    context_parts = []
    for chunk, score in top_chunks:
        context_parts.append(
            f"[{chunk['source']}, chunk {chunk['chunk_id']+1}/{chunk['total_chunks']}]\n{chunk['content']}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    # 4. Генерация
    prompt = f"""Answer the question based on this context.

Context:
{context}

Question: {query}

Answer (be concise):"""
    
    answer = get_completion([{"role": "user", "content": prompt}])
    
    sources = list(set([chunk['source'] for chunk, _ in top_chunks]))
    
    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": [
            (chunk['source'], chunk['chunk_id']+1, f"{score:.3f}") 
            for chunk, score in top_chunks
        ]
    }

# ========== ТЕСТЫ ==========
print("="*60)
print("🤖 RAG СИСТЕМА С CHUNKING")
print("="*60)

# Тест 1
print("\n📝 Вопрос: How does RAG work?")
result = rag_query("How does RAG work?")
print(f"✅ ОТВЕТ:\n{result['answer']}")
print(f"\n📚 Источники: {', '.join(result['sources'])}")
print(f"📊 Chunks: {result['chunks_used']}\n")

# Тест 2
print("📝 Вопрос: What is Python used for?")
result = rag_query("What is Python used for?")
print(f"✅ ОТВЕТ:\n{result['answer']}")
print(f"\n📚 Источники: {', '.join(result['sources'])}\n")

print("="*60)
print("✅ RAG с chunking работает!")
