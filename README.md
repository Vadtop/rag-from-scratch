# RAG From Scratch

Minimal Retrieval-Augmented Generation (RAG) system built from scratch, to understand how everything works under the hood without LangChain and heavy frameworks.

## 🎯 Project Goal

Show practical understanding of RAG fundamentals: embeddings, similarity search, retrieval, answer generation with sources.  
This project is part of my learning path towards an AI/LLM Engineer role.

## 🏗️ Architecture

User Query  
↓  
Embedding Model (`text-embedding-3-small` via OpenRouter)  
↓  
Cosine Similarity Search  
↓  
Top-K Documents Retrieved  
↓  
LLM (`gpt-3.5-turbo`) with Context  
↓  
Generated Answer + Sources

## ✨ Features

- **Manual implementation**: cosine similarity, semantic search, RAG pipeline without LangChain
- **Document loading**: automatic collection of `.txt` files from `documents/` folder
- **Chunking**: splitting long texts into chunks (500 chars with 100 chars overlap)
- **Metadata & sources**: for each chunk we store `source`, `chunk_id`, `total_chunks`
- **Semantic search**: find relevant content by meaning, not by keywords
- **REST API**: FastAPI wrapper around the RAG pipeline for integration with other systems
- **Full RAG pipeline**: end-to-end — from user query to answer with sources

## 🛠️ Tech Stack

- **Python 3.11**
- **FastAPI** (REST API around RAG)
- **OpenRouter API** (embeddings + LLM)
- **NumPy** (vector operations)
- **requests** (HTTP calls to OpenRouter API without `openai` SDK)

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
pip install fastapi uvicorn numpy requests

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

POST /upload — accept raw text or file, split into chunks, compute embeddings and store them in an in-memory “knowledge base”

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
🧪 What I Learned
Core Concepts
Embeddings — converting text into fixed-size vectors (e.g. 1536) to compare meaning instead of raw strings

Cosine similarity — similarity metric between vectors (1 = very similar, 0 = unrelated)

Semantic search — finding documents by meaning using embeddings + similarity search

RAG pipeline — separating retrieval (finding context) and generation (LLM answering)

Chunking & metadata — why we need to split long texts and keep track of where each piece came from

Implementation Details
Cosine similarity implemented manually: dot_product / (norm1 * norm2)

Embeddings and LLM calls are made via OpenRouter API (no openai SDK dependency)

Documents are loaded automatically and split into chunks

Answers include the source (file name and chunk index)

📊 Project Evolution
Current Version (v0.2):

✅ Manual embeddings + similarity search

✅ Document loading + chunking

✅ Source attribution (metadata)

✅ Full RAG pipeline (retrieval + generation)

✅ FastAPI REST API

Planned (v0.3):

Vector DB (Chroma/Qdrant) instead of in-memory storage

Improved chunking (by sentences/paragraphs)

Basic monitoring of requests and responses

Future (v1.0):

Production-ready API (Docker, env configs)

Logging and quality metrics

Web UI (e.g. Streamlit)

Advanced retrieval (hybrid search, reranking)

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