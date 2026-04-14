# RAG AI Agent

**Version:** 3.0 | **Stack:** FastAPI + ChromaDB + DeepSeek + Google Sheets

AI agent with knowledge base (RAG), dialog memory, and Google Sheets logging.

---

## Features

- **RAG** — answers questions based on uploaded documents, doesn't hallucinate
- **AI agent with memory** — remembers dialog history per session (multi-turn)
- **Google Sheets logging** — every Q&A is written to a spreadsheet
- **Two approaches** — hand-built RAG and LangChain implementation for comparison
- **Quality evaluation** — Faithfulness 87.5%, Relevancy 89.6% (LLM-as-judge)
- **Live demo** — deployed on Railway, accessible online

---

## Architecture

```
User Question
↓
Embedding (text-embedding-3-small via OpenRouter)
↓
ChromaDB — semantic search in knowledge base
↓
Top-K relevant chunks
↓
DeepSeek LLM — generate answer from context
↓
Answer + sources + Google Sheets log
```

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/Vadtop/rag-from-scratch.git
cd rag-from-scratch

# Create .env file:
# OPENROUTER_API_KEY=sk-or-...
# GOOGLE_SHEETS_CREDENTIALS_JSON=credentials.json
# GOOGLE_SHEETS_ID=your_sheet_id

docker-compose up --build
```

### Manual

```bash
git clone https://github.com/Vadtop/rag-from-scratch.git
cd rag-from-scratch
pip install -r requirements.txt

# Create .env file (see above)

uvicorn api:app --reload
```

API docs: http://localhost:8000/docs

Live demo: https://rag-from-scratch-production.up.railway.app/chat

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload document to knowledge base |
| POST | `/query` | Ask question (RAG) |
| POST | `/agent` | AI agent with dialog memory |
| DELETE | `/agent/{session_id}` | Clear session history |
| POST | `/query_langchain` | Ask via LangChain RAG |
| GET | `/stats` | Knowledge base statistics |
| DELETE | `/reset` | Clear knowledge base |

---

## Examples

```bash
# Upload document
curl -X POST http://localhost:8000/upload \
  -F "file=@documents/ai_agents_business.txt"

# Ask question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to optimize LLM costs?"}'

# Agent with memory
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "session_id": "user_1"}'

# Continue dialog
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "How to apply this in CRM?", "session_id": "user_1"}'
```

---

## Google Sheets Setup

1. Create Service Account in Google Cloud Console
2. Share your spreadsheet with the service account (Editor)
3. Download `credentials.json`
4. Add to `.env`:
   - `GOOGLE_SHEETS_CREDENTIALS_JSON=credentials.json`
   - `GOOGLE_SHEETS_ID=` (ID from spreadsheet URL)

The agent auto-creates a "RAG Log" sheet and logs all queries there.

---

## Quality Metrics

```bash
python evaluate.py
```

- **Faithfulness:** 87.5% — answer is based on documents, not hallucinated
- **Relevancy:** 89.6% — correct chunks were retrieved
- **Overall:** 88.5%

---

## Tech Stack

- Python 3.11, FastAPI, uvicorn
- ChromaDB (vector DB, HNSW indexing)
- OpenRouter API (DeepSeek + embeddings)
- LangChain (alternative RAG implementation)
- gspread (Google Sheets integration)
- Docker + docker-compose

---

## Project Evolution

- **v1.0** — hand-built RAG, in-memory storage
- **v2.0** — ChromaDB, persistent storage
- **v2.1** — LangChain + auto quality evaluation
- **v3.0** — AI agent with dialog memory + Google Sheets logging

---

## Author

[Vadim Titov](https://github.com/Vadtop)
