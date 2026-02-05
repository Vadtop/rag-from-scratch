from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma  # ← ИЗМЕНЕНО
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader  # ← ИЗМЕНЕНО
from langchain.text_splitter import CharacterTextSplitter

# ========== НАСТРОЙКА API ==========

import os
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"

# ========== ЗАГРУЗКА ДОКУМЕНТОВ ==========

print("📂 Загружаю документы...")

loader = DirectoryLoader(
    "documents/",
    glob="*.txt",
    loader_cls=TextLoader
)

documents = loader.load()
print(f"✅ Загружено документов: {len(documents)}")

# ========== СОЗДАНИЕ EMBEDDINGS + VECTOR STORE ==========

print("\n⏳ Создаю embeddings...")

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

print("✅ Vector store готов!")

# ========== СОЗДАНИЕ LLM ==========

llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    temperature=0
)

# ========== СОЗДАНИЕ RAG CHAIN ==========

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Простейший тип: все документы в контекст
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)

# ========== ТЕСТЫ ==========

print("\n" + "="*60)
print("🤖 RAG СИСТЕМА (LANGCHAIN)")
print("="*60)

# Тест 1
print("\n📝 Вопрос: How does RAG work?")
answer = qa_chain.run("How does RAG work?")
print(f"✅ ОТВЕТ:\n{answer}\n")

# Тест 2
print("📝 Вопрос: What is Python used for?")
answer = qa_chain.run("What is Python used for?")
print(f"✅ ОТВЕТ:\n{answer}\n")

# Тест 3
print("📝 Вопрос: What are vector databases?")
answer = qa_chain.run("What are vector databases used for?")
print(f"✅ ОТВЕТ:\n{answer}\n")

print("="*60)
