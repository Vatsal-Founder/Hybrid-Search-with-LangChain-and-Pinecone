# Hybrid Search RAG — LangChain + Pinecone 

An end‑to‑end **Hybrid Search** RAG app for **research papers** you upload. Built with **LangChain** and **Pinecone** and implemented as a **Streamlit** app.

* 🔎 **Hybrid retrieval**: combines **dense embeddings** with **sparse BM25** signals for high‑recall, high‑precision search.
* 📄 **PDF ingestion**: drop in a paper, chunk → embed → index.
* 🧠 **RAG pipeline**: retrieve relevant chunks and generate grounded answers.
* 🗂️ **Pinecone vector DB**: fast, scalable, persistent.

> Repo layout highlights: `searchapp.py` (Streamlit app), `requirements.txt`, and `bm25_values.json` (precomputed BM25 stats).

Link:
---

## Features

* **Upload one or more PDFs** and build a hybrid index in Pinecone.
* **Ask questions** and get answers with cited passages.
* **BM25 + embeddings** hybrid scoring for better relevance on technical text.
* **Configurable** chunking, top‑k, and model/encoder settings via env.

---

## Architecture

```
[User (Streamlit UI)]
      │  upload PDF(s), ask questions
      ▼
[Loader]  →  [Text Splitter]  →  [Embeddings (dense)] ┐
                                          ├─ upsert → [Pinecone Hybrid Index]
[Sparse Encoder (BM25)]  ─────────────────┘
      │                                 │
      ▼                                 ▼
  user query  ───────────────▶  [Hybrid Retriever] ─▶ [LLM (answer + sources)]
```

---

## Project Structure

```
.
├── searchapp.py          # Streamlit app (ingest + chat)
├── requirements.txt      # Python deps
├── bm25_values.json      # saved BM25 stats (optional)
└── README.md             # this file
```

---

## Requirements

* Python **3.10+**
* **Pinecone** account & API key
* One embeddings provider (e.g., OpenAI or Hugging Face)

---

## Quick Start

### 1) Install

```bash
git clone https://github.com/Vatsal-Founder/Hybrid-Search-with-LangChain-and-Pinecone.git
cd Hybrid-Search-with-LangChain-and-Pinecone
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # create if missing
```

### 2) Configure environment

Create `.env` with your keys and defaults. Minimal example:

```ini
# Pinecone
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=hybrid-rag
PINECONE_CLOUD=aws           # serverless: aws | gcp | azure
PINECONE_REGION=us-east-1    # serverless region

# Embeddings (choose one)
OPENAI_API_KEY=your_openai_key
EMBEDDINGS_MODEL=text-embedding-3-small   # or HF model name
# HF_API_TOKEN=your_huggingface_token

# Retrieval

BM25_PATH=./bm25_values.json   # optional; auto-built if empty
```

### 3) Run the app

```bash
streamlit run searchapp.py
```

Open [http://localhost:8501](http://localhost:8501), upload a PDF, click **Index** (if shown), and start asking questions.

---

## Using the App

1. **Upload research paper(s)** in PDF.
2. The app **splits** text, computes **dense embeddings** and **BM25 sparse values**, and **upserts** to Pinecone.
3. Ask a question. The **Hybrid Retriever** mixes dense + sparse scores to fetch the most relevant chunks.
4. The LLM composes an answer and returns **citations/snippets** from the source.

> Tip: Hybrid search shines on technical papers—keep chunk size around 800–1200 with \~120–200 overlap for stable recall.

---

## Deployment (optional)

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "searchapp.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

Build & run:

```bash
docker build -t hybrid-rag .
docker run -p 8501:8501 --env-file .env hybrid-rag
```

### PaaS

* Command: `streamlit run searchapp.py --server.port $PORT --server.address 0.0.0.0`
* Set env vars in the platform dashboard; persist index in Pinecone.

---

## Configuration Tips

* **Embeddings**: start with a small, fast model; upgrade for nuanced semantics.
* **BM25**: if `bm25_values.json` exists, it will be loaded; otherwise the app can fit and save new stats.
* **Alpha/balancing**: if surfaced in the UI, tune the dense–sparse balance per corpus.

---

