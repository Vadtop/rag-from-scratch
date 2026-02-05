from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()  # подхватит OPENROUTER_API_KEY из .env

API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# Функция: превратить текст в вектор чисел
def get_embedding(text):
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# Тест: берём 3 фразы
text1 = "Python is a programming language"
text2 = "Java is a programming language"
text3 = "I love pizza"


# Получаем векторы для каждой фразы
emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)


# Смотрим что получилось
print(f"Размер вектора: {len(emb1)}")
print(f"Первые 5 чисел вектора: {emb1[:5]}")


# ========== Функция для сравнения векторов ==========


def cosine_similarity(vec1, vec2):
    """Считает похожесть двух векторов (от -1 до 1)"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


# Сравниваем тексты
print("\n=== СРАВНЕНИЕ ТЕКСТОВ ===")


sim_python_java = cosine_similarity(emb1, emb2)
print(f"Python vs Java: {sim_python_java:.3f}")


sim_python_pizza = cosine_similarity(emb1, emb3)
print(f"Python vs Pizza: {sim_python_pizza:.3f}")


sim_java_pizza = cosine_similarity(emb2, emb3)
print(f"Java vs Pizza: {sim_java_pizza:.3f}")


# ========== ЗАГРУЗКА ДОКУМЕНТОВ ИЗ ФАЙЛОВ ==========


def load_documents_from_folder(folder_path="documents"):
    """Загружает все .txt файлы из папки"""
    documents = []
    
    print(f"\n📂 Загружаю документы из '{folder_path}/'...")
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append({
                        "content": content,
                        "source": filename
                    })
                    print(f"  ✅ {filename}")
    
    print(f"📊 Загружено документов: {len(documents)}\n")
    return documents


# ========== ПОИСКОВИК ПО ДОКУМЕНТАМ ==========


# Загружаем документы из файлов
documents = load_documents_from_folder("documents")


print("=== БАЗА ДОКУМЕНТОВ ===")
for i, doc in enumerate(documents, 1):
    print(f"{i}. [{doc['source']}] {doc['content'][:80]}...")


# Получаем embeddings для документов (в реальности делается 1 раз)
print("\n⏳ Получаем embeddings...")
doc_embeddings = [get_embedding(doc["content"]) for doc in documents]
print("✅ Готово!")


# Функция поиска
def search(query, top_k=2):
    """Ищет top_k наиболее похожих документов"""
    print(f"\n🔍 Запрос: '{query}'")
    
    # Получаем embedding для запроса
    query_emb = get_embedding(query)
    
    # Считаем similarity для каждого документа
    results = []
    for doc_obj, doc_emb in zip(documents, doc_embeddings):
        sim = cosine_similarity(query_emb, doc_emb)
        results.append((doc_obj, sim))
    
    # Сортируем по убыванию
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Возвращаем top_k
    return results[:top_k]


# ТЕСТЫ
print("\n" + "="*50)
print("ТЕСТ ПОИСКОВИКА")
print("="*50)


# Тест 1
results = search("Tell me about Python")
for i, (doc_obj, score) in enumerate(results, 1):
    print(f"  {i}. [{score:.3f}] {doc_obj['content'][:60]}... (источник: {doc_obj['source']})")


# ============= ГЕНЕРАЦИЯ ОТВЕТА ЧЕРЕЗ LLM ================


def generate_answer(query, relevant_docs):
    """Генерирует ответ используя найденные документы"""

    # Формируем контекст из найденных документов
    context = "\n".join([f"- {doc_obj['content']}" for doc_obj, score in relevant_docs])

    # Формируем промт
    prompt = f"""На основе этой информации ответь на вопрос.

Контекст:
{context}

Вопрос: {query}

Ответ (кратко и под делу):"""
    
    # Запрос к LLM через OpenRouter
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content


# ========== ПОЛНЫЙ RAG (поиск + генерация) ==========


def rag_pipeline(query, top_k=2):
    """Полный RAG: находит документы и генерирует ответ"""
    
    print(f"\n{'='*60}")
    print(f"🤖 RAG СИСТЕМА")
    print(f"{'='*60}")
    print(f"📝 Вопрос: {query}\n")
    
    # Шаг 1: Поиск релевантных документов
    print("🔍 Ищу релевантные документы...")
    relevant_docs = search(query, top_k=top_k)
    
    print("\n📚 Найденные документы:")
    for i, (doc_obj, score) in enumerate(relevant_docs, 1):
        print(f"  {i}. [{score:.3f}] {doc_obj['content'][:60]}...")
        print(f"      📄 Источник: {doc_obj['source']}")
    
    # Шаг 2: Генерация ответа
    print("\n💭 Генерирую ответ...")
    answer = generate_answer(query, relevant_docs)
    
    # Показываем источники
    sources = [doc_obj['source'] for doc_obj, score in relevant_docs]
    sources_text = ", ".join(sources)

    print(f"\n✅ ОТВЕТ:\n{answer}")
    print(f"\n📚 Источники: {sources_text}")
    print(f"{'='*60}\n")
    
    return answer


# ========== ТЕСТЫ ПОЛНОГО RAG ==========


print("\n\n" + "🚀"*30)
print("ТЕСТИРУЕМ ПОЛНЫЙ RAG")
print("🚀"*30 + "\n")


# Тест 1: Про RAG
rag_pipeline("How does RAG work?")


# Тест 2: Про vector DB
rag_pipeline("What are vector databases used for?")


# Тест 3: Про OpenRouter
rag_pipeline("What is OpenRouter?")
