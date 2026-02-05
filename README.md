# RAG From Scratch

**Version:** 2.0 | **Status:** Production-ready with ChromaDB

Minimal Retrieval-Augmented Generation (RAG) system built from scratch, to understand how everything works under the hood without LangChain and heavy frameworks.

## 🎯 Project Goal

Show practical understanding of RAG fundamentals: embeddings, similarity search, retrieval, answer generation with sources.  
This project is part of my learning path towards an AI/LLM Engineer role.

## 🏗️ Architecture

User Query  
↓  
Embedding Model (`text-embedding-3-small` via OpenRouter)  
↓  
**ChromaDB Vector Store** (HNSW similarity search)  
↓  
Top-K Documents Retrieved  
↓  
LLM (`gpt-3.5-turbo`) with Context  
↓  
Generated Answer + Sources

## ✨ Features

- **ChromaDB integration**: Persistent vector storage with HNSW indexing for fast similarity search
- **Manual implementation**: Full RAG pipeline understanding without heavy frameworks
- **Document loading**: automatic collection of `.txt` files from `documents/` folder
- **Chunking**: splitting long texts into chunks (500 chars with 100 chars overlap)
- **Metadata & sources**: for each chunk we store `source`, `chunk_id`, `total_chunks`
- **Semantic search**: find relevant content by meaning, not by keywords
- **REST API**: FastAPI wrapper around the RAG pipeline for integration with other systems
- **Full RAG pipeline**: end-to-end — from user query to answer with sources

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
git clone https://github.com/YOUR_USERNAME/rag-from-scratch.git
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
🚀 Usage (CLI demo)
Simple script to test RAG locally without API:

bash
python step1_embeddings.py
What it does:

Loads .txt files from documents/

Computes embeddings

Runs similarity search

Calls LLM and prints answer + sources

🌐 REST API
api.py exposes a FastAPI service on top of the RAG pipeline:

POST /upload — accept raw text or file, split into chunks, compute embeddings and store them in ChromaDB vector database

POST /query — accept a question, find top-K relevant chunks and generate an answer with sources

GET /stats — show how many documents and chunks are currently loaded

DELETE /reset — clear the knowledge base

Run API:

bash
uvicorn api:app --reload
Open interactive docs at:
http://localhost:8000/docs

This allows using RAG as a standalone service that can be connected to chat-bots, frontends or internal tools.

📂 Add Your Own Documents
Create .txt files in the documents/ folder.

Run the script or API — documents will be automatically loaded and chunked.

Ask questions related to your content.

Example:

text
# documents/python.txt
Python is a high-level programming language...

# documents/machine_learning.txt
Machine learning is a subset of AI...
python
# CLI demo
rag_pipeline("What is Python used for?")
Sample output:

text
✅ ANSWER:
Python is widely used for web development, data science,
automation, and artificial intelligence.

📚 Sources: python.txt (chunk 1/2)

## 🆕 What's New in v2.0

### ChromaDB Integration
- ✅ **Persistent storage**: Embeddings saved to disk (`./chroma_db/`), survive restarts
- ✅ **HNSW indexing**: Fast similarity search even with thousands of documents
- ✅ **Metadata support**: Track source, chunk_id, total_chunks for each embedding
- ✅ **Production-ready**: No more in-memory lists, scalable architecture

### Migration from v1.0
**v1.0** used in-memory Python list to store embeddings:
- Lost all data on restart
- Slow linear search through all embeddings
- Good for learning, not for production

**v2.0** uses ChromaDB:
- Data persists on disk
- Optimized vector search with indexing
- Ready for real-world usage

To see v1.0 code: `git checkout v1.0`


## 🧪 What I Learned

### Core Concepts
- Embeddings — converting text into fixed-size vectors
- Cosine similarity — similarity metric between vectors
- Semantic search — finding documents by meaning
- RAG pipeline — separating retrieval and generation
- Chunking & metadata — splitting texts and tracking sources
- **Vector databases** — specialized storage for embeddings with fast similarity search

### Implementation Details
- ~~Cosine similarity implemented manually~~ → **Replaced with ChromaDB HNSW indexing**
- Embeddings and LLM calls via OpenRouter API
- Documents loaded and chunked automatically
- **Persistent storage** with ChromaDB (survives restarts)
- Answers include source attribution

## 📊 Project Evolution

**Current Version (v2.0):**
✅ ChromaDB vector database integration  
✅ Persistent storage (chroma_db/ folder)  
✅ HNSW indexing for fast search  
✅ Production-ready architecture  
✅ Full RAG pipeline with metadata  
✅ FastAPI REST API  

**Previous Version (v1.0):**
✅ Manual embeddings + cosine similarity  
✅ In-memory storage  
✅ Basic RAG pipeline  

**Planned (v2.1):**
- [ ] Improved chunking (by sentences/paragraphs)
- [ ] Basic monitoring of requests and responses
- [ ] Multiple document formats (PDF, DOCX)

**Future (v3.0):**
- [ ] Docker deployment
- [ ] Logging and quality metrics
- [ ] Web UI (Streamlit)
- [ ] Hybrid search (keyword + semantic)


💡 Why This Approach?
Starting with plain Python and minimal dependencies, then moving to frameworks. This gives:

Clear understanding of how RAG works inside

Ability to debug and optimize the pipeline for specific products

Confidence in technical interviews when asked “what’s inside LangChain?”

“Junior uses libraries. Middle understands what happens under the hood.”

🎓 Interview Readiness
Based on this project I can:

Explain the difference between keyword search and semantic search

Describe the RAG pipeline and why chunking is needed

Show working RAG + API code

Discuss where it makes sense to plug in a vector DB and monitoring

📝 Notes
This is a learning project, not a full production solution

API keys are not included in the repo — use your own via environment variables

Built as part of an intensive learning path to transition into AI/LLM Engineering

📧 Contact
Built by Vadim Titov as part of transition to an AI/LLM Engineer role.
Focus areas: RAG, automation, AI assistants for customer support.