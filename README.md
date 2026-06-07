# Local RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with:

- FastAPI
- ChromaDB
- Sentence Transformers
- Ollama
- Llama 3.2 3B

## Features

- Upload PDF or TXT documents
- Automatic chunking and embedding
- Semantic search using vector embeddings
- Local LLM inference with Ollama
- REST API using FastAPI

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/local-rag-chatbot.git
cd local-rag-chatbot

pip install -r requirements.txt
```

Install Ollama:

```bash
ollama pull llama3.2:3b
```

Run the API:

```bash
uvicorn main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```
GET /health
```

### Upload Document

```
POST /ingest
```

Accepts:

- PDF
- TXT

### Chat

```
POST /chat
```

Example:

```json
{
    "question": "What is machine learning?"
}
```

## Architecture

```
Document
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Similarity Search
    ↓
Prompt Construction
    ↓
Llama 3.2 (Ollama)
    ↓
Answer
```

## Tech Stack

- Python
- FastAPI
- ChromaDB
- Sentence Transformers
- Ollama
- Llama3.2:3B