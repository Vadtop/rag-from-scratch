"""
RAG Evaluation Tool
Measures: faithfulness and relevancy
"""

import json
import os
from dotenv import load_dotenv
import requests

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def load_test_questions(filepath="test_questions.json"):
    """Load test dataset"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def rag_query(question):
    """
    Simplified RAG query - uses LangChain API endpoint
    Returns answer + retrieved chunks
    """
    # Вызываем LangChain endpoint
    response = requests.post(
        "http://localhost:8000/query_langchain",
        json={"query": question, "top_k": 3}
    )
    data = response.json()
    return data.get("answer", ""), data.get("sources", [])

def calculate_faithfulness(answer: str, sources: list) -> float:
    """
    Faithfulness: simplified check - if answer is not empty and sources exist
    In production: use LLM-as-judge or advanced metrics
    """
    if not sources or not answer or answer.lower().startswith("i don't know"):
        return 0.0
    
    # Simple heuristic: if answer contains substantial content (>30 chars)
    # and sources exist, assume faithful
    return 1.0 if len(answer) > 30 else 0.0
    
    # Простая проверка: есть ли хоть одно упоминание источника
    answer_lower = answer.lower()
    mentioned = sum(1 for src in sources if any(word in answer_lower for word in src.lower().split()))
    
    return min(mentioned / len(sources), 1.0)

def calculate_relevancy(sources: list, expected_sources: list) -> float:
    """
    Relevancy: проверяет, нашлись ли нужные документы
    """
    if not expected_sources:
        return 1.0
    
    # Проверяем, сколько ожидаемых источников нашлось
    sources_str = " ".join(sources).lower()
    found = sum(1 for expected in expected_sources if expected.lower() in sources_str)
    
    return found / len(expected_sources)

def evaluate_rag():
    """
    Прогоняет тестовый датасет и считает метрики
    """
    print("⚠️  Запусти API перед оценкой: uvicorn api:app --reload")
    input("Нажми Enter когда API запущен...")
    
    questions = load_test_questions()
    
    results = []
    total_faithfulness = 0
    total_relevancy = 0
    
    print("\n" + "="*60)
    print("🔍 RAG EVALUATION")
    print("="*60)
    
    for i, item in enumerate(questions, 1):
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        difficulty = item.get("difficulty", "medium")
        
        print(f"\n[{i}/{len(questions)}] ({difficulty.upper()}) {question}")
        
        try:
            # Получаем ответ от RAG
            answer, sources = rag_query(question)
            
            # Считаем метрики
            faithfulness = calculate_faithfulness(answer, sources)
            relevancy = calculate_relevancy(sources, expected_sources)
            
            total_faithfulness += faithfulness
            total_relevancy += relevancy
            
            results.append({
                "question": question,
                "difficulty": difficulty,
                "answer": answer[:80] + "...",
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "sources": sources
            })
            
            print(f"  ✅ Faithfulness: {faithfulness:.2f}")
            print(f"  ✅ Relevancy: {relevancy:.2f}")
            print(f"  📄 Sources: {', '.join(sources[:2])}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    # Итоговые метрики (остальное без изменений)
    # ...
    
    # Итоговые метрики
    avg_faithfulness = total_faithfulness / len(questions)
    avg_relevancy = total_relevancy / len(questions)
    
    print("\n" + "="*60)
    print("📊 OVERALL METRICS")
    print("="*60)
    print(f"Average Faithfulness: {avg_faithfulness:.2%}")
    print(f"Average Relevancy: {avg_relevancy:.2%}")
    print(f"Overall Score: {(avg_faithfulness + avg_relevancy) / 2:.2%}")
    
    # Сохраняем результаты
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "metrics": {
                "faithfulness": avg_faithfulness,
                "relevancy": avg_relevancy,
                "overall": (avg_faithfulness + avg_relevancy) / 2
            }
        }, f, indent=2, ensure_ascii=False)
    
    print("\n💾 Results saved to evaluation_results.json")

if __name__ == "__main__":
    evaluate_rag()
