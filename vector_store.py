import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, collection_name="rag_documents"):
        # Создаем клиент с сохранением на диск
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Получаем или создаем коллекцию
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    
    def add_chunk(self, chunk_id, content, embedding, metadata):
        """Добавить один chunk"""
        self.collection.add(
            ids=[chunk_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    def search(self, query_embedding, top_k=3):
        """Поиск похожих chunks"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
    
    def count(self):
        """Количество chunks в базе"""
        return self.collection.count()
    
    def clear(self):
        """Очистить базу"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name
        )
    
    def get_all_sources(self):
        """Получить список всех источников"""
        all_data = self.collection.get()
        if not all_data['metadatas']:
            return []
        sources = set([meta['source'] for meta in all_data['metadatas']])
        return list(sources)
