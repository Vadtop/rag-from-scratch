# RAG From Scratch

**Version:** 2.1 | **Status:** Production-ready with ChromaDB + LangChain + Evaluation

Retrieval-Augmented Generation (RAG) system with dual implementation: 
from-scratch approach and LangChain integration for comparison.

## 🎯 Project Goal

Production-ready RAG system demonstrating both manual implementation and 
framework-based approach with automated quality evaluation.  

## 🏗️ Architecture

```
User Query
↓
Embedding Model (text-embedding-3-small via OpenRouter)
↓
ChromaDB Vector Store (HNSW similarity search)
↓
Top-K Documents Retrieved
↓
LLM (gpt-3.5-turbo) with Context
↓
Generated Answer + Sources
```

## ✨ Features

- **ChromaDB integration**: Persistent vector storage with HNSW indexing for fast similarity search
- **Manual implementation**: Full RAG pipeline understanding without heavy frameworks
- **Document loading**: automatic collection of `.txt` files from `documents/` folder
- **Chunking**: splitting long texts into chunks (500 chars with 100 chars overlap)
- **Metadata & sources**: for each chunk we store `source`, `chunk_id`, `total_chunks`
- **Semantic search**: find relevant content by meaning, not by keywords
- **REST API**: FastAPI wrapper around the RAG pipeline for integration with other systems
- **Full RAG pipeline**: end-to-end — from user query to answer with sources
- **Automated evaluation**: Faithfulness and Relevancy metrics with LLM-as-judge

## 🛠️ Tech Stack

- **Python 3.11**
- **FastAPI** (REST API around RAG)
- **ChromaDB** (vector database for persistent embeddings storage)
- **OpenRouter API** (embeddings + LLM)
- **NumPy** (vector operations)
- **requests** (HTTP calls to OpenRouter API)

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/Vadtop/rag-from-scratch.git
cd rag-from-scratch

# (optional) create venv
python -m venv venv
# Windows:
# venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # if you have it
# or minimal:
pip install fastapi uvicorn numpy requests python-dotenv chromadb

# Set up API key (OpenRouter)
# Linux / macOS:
export OPENROUTER_API_KEY="your-key-here"
# Windows (cmd):
# set OPENROUTER_API_KEY=your-key-here
```

## 🚀 Usage (CLI demo)

Simple script to test RAG locally without API:

```bash
python step1_embeddings.py
```

What it does:
- Loads .txt files from documents/
- Computes embeddings
- Runs similarity search
- Calls LLM and prints answer + sources

## 🌐 REST API

api.py exposes a FastAPI service on top of the RAG pipeline:

- **POST /upload** — accept raw text or file, split into chunks, compute embeddings and store them in ChromaDB vector database
- **POST /query** — accept a question, find top-K relevant chunks and generate an answer with sources
- **POST /upload_langchain** — upload documents to LangChain RAG
- **POST /query_langchain** — query using LangChain RAG
- **GET /stats** — show how many documents and chunks are currently loaded
- **DELETE /reset** — clear the knowledge base

Run API:

```bash
uvicorn api:app --reload
```

Open interactive docs at: http://localhost:8000/docs

This allows using RAG as a standalone service that can be connected to chat-bots, frontends or internal tools.

## 📂 Add Your Own Documents

1. Create .txt files in the documents/ folder
2. Run the script or API — documents will be automatically loaded and chunked
3. Ask questions related to your content

Example:

```python
# documents/python.txt
Python is a high-level programming language...
```

```python
# documents/machine_learning.txt
Machine learning is a subset of AI...
```

CLI demo:

```python
rag_pipeline("What is Python used for?")
```

Sample output:

```
✅ ANSWER:
Python is widely used for web development, data science,
automation, and artificial intelligence.

📚 Sources: python.txt (chunk 1/2)
```

## 🆕 What's New in v2.1

### LangChain Integration

✅ Two approaches in one project: Manual RAG + LangChain RAG side-by-side  
✅ Fast prototyping: LangChain endpoints for rapid development  
✅ Comparison ready: Test both approaches with same queries

### ChromaDB Integration (v2.0)

✅ Persistent storage: Embeddings saved to disk (./chroma_db/), survive restarts  
✅ HNSW indexing: Fast similarity search even with thousands of documents  
✅ Metadata support: Track source, chunk_id, total_chunks for each embedding  
✅ Production-ready: No more in-memory lists, scalable architecture

### Migration from v1.0

v1.0 used in-memory Python list to store embeddings:
- Lost all data on restart
- Slow linear search through all embeddings
- Good for prototyping, not for production

v2.0+ uses ChromaDB:
- Data persists on disk
- Optimized vector search with indexing
- Ready for real-world usage

To see v1.0 code: `git checkout v1.0`

## 🧪 Evaluation System

### Metrics

- **Faithfulness**: 87.5% — checks if answer is grounded in retrieved documents (no hallucinations)
- **Relevancy**: 89.6% — measures retrieval accuracy (correct documents found)
- **Overall Score**: 88.5%

### How It Works

```bash
python evaluate.py
```

Automated evaluation using LLM-as-judge approach:
- Test dataset with 8 questions (easy/medium/hard difficulty)
- Each question tested against RAG system
- LLM evaluates if answer is faithful and relevant
- Results saved to evaluation_results.json

### Test Coverage

- **Easy questions**: Direct facts from documents (100% accuracy)
- **Medium questions**: Require reasoning across chunks (80% accuracy)
- **Hard questions**: Out-of-scope queries (correctly responds "I don't know")

### Why This Matters

- Tracks quality regressions when modifying RAG pipeline
- Provides objective metrics for comparing different approaches
- Essential for production systems (monitoring answer quality)


## 🔄 Two Approaches Comparison

This project demonstrates two ways to build RAG:

### 1️⃣ Manual RAG (/query)

**Pros:**
- Full control over every step
- Easy to debug and customize
- Understand how everything works

**Cons:**
- More code to write (~200 lines)
- Need to handle errors manually

**Use when:** Custom logic, learning, full flexibility needed

### 2️⃣ LangChain RAG (/query_langchain)

**Pros:**
- Fast development (~20 lines)
- Built-in error handling
- Production-ready patterns

**Cons:**
- Less control (black box)
- Framework dependency

**Use when:** Quick MVP, standard use case, time pressure

## 📊 Live Comparison

Both endpoints work in parallel:

```python
# Manual approach
POST /upload → /query

# LangChain approach
POST /upload_langchain → /query_langchain
```

**Key insight:** Understanding fundamentals (manual) + knowing frameworks (LangChain) = strong engineer

## 💡 Why This Approach?

Starting with plain Python and minimal dependencies, then moving to frameworks. This gives:

- Clear understanding of how RAG works inside
- Ability to debug and optimize the pipeline for specific products


## 🔧 Technical Highlights

### Core Concepts

- **Embeddings** — converting text into fixed-size vectors
- **Cosine similarity** — similarity metric between vectors
- **Semantic search** — finding documents by meaning
- **RAG pipeline** — separating retrieval and generation
- **Chunking & metadata** — splitting texts and tracking sources
- **Vector databases** — specialized storage for embeddings with fast similarity search
- **LangChain framework** — rapid prototyping vs manual implementation trade-offs
- **RAG evaluation** — Faithfulness and Relevancy metrics, LLM-as-judge approach

### Implementation Details

- Cosine similarity implemented manually → Replaced with ChromaDB HNSW indexing
- Embeddings and LLM calls via OpenRouter API
- Documents loaded and chunked automatically
- Persistent storage with ChromaDB (survives restarts)
- Answers include source attribution
- Two parallel implementations for comparison
- Automated testing with evaluation metrics

## 📊 Project Evolution

This project evolved through multiple iterations:

**v1.0 → v2.0 → v2.1**

- **v1.0**: Manual RAG with in-memory storage (prototype phase)
- **v2.0**: Added ChromaDB for production-ready persistent storage
- **v2.1**: Integrated LangChain + automated evaluation system

### Current features (v2.1):

✅ ChromaDB vector database with HNSW indexing  
✅ Persistent storage on disk  
✅ FastAPI REST API  
✅ Two parallel implementations (Manual + LangChain)  
✅ Full metadata tracking and source attribution  
✅ Automated evaluation with Faithfulness and Relevancy metrics

### What's next (v2.2):

⬜ Improved chunking strategies  
⬜ Basic monitoring and metrics  
⬜ Multiple document formats (PDF, DOCX)


## 📧 Contact
 
GitHub: @Vadtop

