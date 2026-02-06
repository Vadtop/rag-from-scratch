# step2_langchain.py
"""
LangChain-based RAG implementation for comparison with manual approach.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

# ========== КОНФИГУРАЦИЯ ==========
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"
PERSIST_DIR = "./chroma_langchain_db"

# ========== ИНИЦИАЛИЗАЦИЯ ==========

class LangChainRAG:
    def __init__(self):
        """Инициализация LangChain RAG системы"""
        
        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL
        )
        
        # LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL
        )
        
        # Попытка загрузить существующий vectorstore
        try:
            self.vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=self.embeddings
            )
            print(f"✅ Загружен существующий vectorstore ({self.vectorstore._collection.count()} документов)")
        except:
            self.vectorstore = None
            print("⚠️ Vectorstore не найден. Используйте load_documents() для создания.")
        
        # QA chain
        if self.vectorstore:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
            )
    
    def load_documents(self, directory="documents/"):
        """Загрузить документы и создать vectorstore"""
        
        print(f"📂 Загружаю документы из {directory}...")
        
        # Загрузка
        loader = DirectoryLoader(
            directory,
            glob="*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"✅ Загружено документов: {len(documents)}")
        
        # Chunking (как в manual RAG для сравнения)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Создано chunks: {len(chunks)}")
        
        # Создание vectorstore
        print("⏳ Создаю embeddings...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIR
        )
        print("✅ Vector store готов и сохранен на диск!")
        
        # Создание QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        
        return len(chunks)
    
    def query(self, question: str) -> dict:
        """Задать вопрос RAG системе"""
        
        if not self.qa_chain:
            return {
                "answer": "Vectorstore not initialized. Please load documents first.",
                "sources": []
            }
        
        # Получение ответа
        answer = self.qa_chain.run(question)
        
        # Получение источников
        docs = self.vectorstore.similarity_search(question, k=3)
        sources = list(set([doc.metadata.get('source', 'unknown') for doc in docs]))
        
        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(docs)
        }


# ========== ТЕСТЫ (если запускаешь напрямую) ==========

if __name__ == "__main__":
    print("="*60)
    print("🤖 LANGCHAIN RAG СИСТЕМА")
    print("="*60)
    
    # Инициализация
    rag = LangChainRAG()
    
    # Загрузка документов (если еще не загружены)
    if rag.vectorstore is None or rag.vectorstore._collection.count() == 0:
        rag.load_documents()
    
    # Тесты
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ")
    print("="*60)
    
    questions = [
        "How does RAG work?",
        "What is Python used for?",
        "What are vector databases?"
    ]
    
    for q in questions:
        print(f"\n📝 Вопрос: {q}")
        result = rag.query(q)
        print(f"✅ ОТВЕТ:\n{result['answer']}")
        print(f"📚 Источники: {', '.join(result['sources'])}")
        print("-"*60)
