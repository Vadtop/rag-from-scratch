# RAG AI Agent

**Version:** 3.0 | **Stack:** FastAPI + ChromaDB + DeepSeek + Google Sheets

AI-агент с базой знаний (RAG), памятью диалога и логированием в Google Sheets.

## Что умеет

- **RAG** — отвечает на вопросы на основе загруженных документов, не выдумывает
- **AI-агент с памятью** — помнит историю диалога в рамках сессии (multi-turn)
- **Google Sheets логирование** — каждый вопрос и ответ пишется в таблицу
- **Два подхода** — ручная реализация RAG и LangChain для сравнения
- **Автооценка качества** — Faithfulness 87.5%, Relevancy 89.6% (LLM-as-judge)

## Архитектура

```
Вопрос пользователя
↓
Embedding (text-embedding-3-small via OpenRouter)
↓
ChromaDB — семантический поиск по базе знаний
↓
Top-K релевантных чанков
↓
DeepSeek LLM — генерация ответа на основе контекста
↓
Ответ + источники + запись в Google Sheets
```

## Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/upload` | Загрузить документ в базу знаний |
| POST | `/query` | Задать вопрос (RAG) |
| POST | `/agent` | AI-агент с памятью диалога |
| DELETE | `/agent/{session_id}` | Очистить историю сессии |
| POST | `/upload_langchain` | Загрузить через LangChain |
| POST | `/query_langchain` | Спросить через LangChain |
| GET | `/stats` | Статистика базы знаний |
| DELETE | `/reset` | Очистить базу |

## Быстрый старт

```bash
git clone https://github.com/Vadtop/ai-agent-rag.git
cd ai-agent-rag
pip install -r requirements.txt

# Создать .env файл:
# OPENROUTER_API_KEY=sk-or-...
# GOOGLE_SHEETS_CREDENTIALS_JSON=credentials.json
# GOOGLE_SHEETS_ID=твой_id_таблицы

uvicorn api:app --reload
```

Документация: http://localhost:8000/docs

## Примеры запросов

```bash
# Загрузить документ
curl -X POST http://localhost:8000/upload \
  -F "file=@documents/ai_agents_business.txt"

# Задать вопрос
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Как оптимизировать стоимость LLM?"}'

# Агент с памятью диалога
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Что такое RAG?", "session_id": "user_1"}'

# Продолжить диалог
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "А как это применить в CRM?", "session_id": "user_1"}'
```

## Google Sheets настройка

1. Создать Service Account в Google Cloud Console
2. Дать доступ к таблице (Editor)
3. Скачать `credentials.json`
4. Добавить в `.env`:
   - `GOOGLE_SHEETS_CREDENTIALS_JSON=credentials.json`
   - `GOOGLE_SHEETS_ID=` (ID из URL таблицы)

Агент автоматически создаст лист "RAG Log" и будет писать туда все запросы.

## Стек

- Python 3.11, FastAPI, uvicorn
- ChromaDB (векторная БД, HNSW индексирование)
- OpenRouter API (DeepSeek + embeddings)
- LangChain (альтернативная реализация)
- gspread (Google Sheets интеграция)

## Метрики качества

```bash
python evaluate.py
```

- Faithfulness: 87.5% — ответ основан на документах, не выдуман
- Relevancy: 89.6% — найдены правильные чанки
- Overall: 88.5%

## Эволюция проекта

- **v1.0** — ручной RAG, хранение в памяти
- **v2.0** — ChromaDB, персистентное хранение
- **v2.1** — LangChain + автооценка качества
- **v3.0** — AI-агент с памятью диалога + Google Sheets логирование

---

GitHub: [@Vadtop](https://github.com/Vadtop)
