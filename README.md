# On Device RAG Assistant (LangChain + Ollama + FAISS)

A local (on device) Retrieval Augmented Generation pipeline that answers questions **grounded in your own documents** instead of guessing.  
It ingests files, chunks + embeds them, stores embeddings in a vector index, retrieves the most relevant passages for a query, and generates an answer using **Ollama** (local LLM) with the retrieved context to reduce hallucinations.

---

## Why this project

LLMs are impressive, but they can still hallucinate when the real answer is buried inside PDFs, docs, or internal notes.  
This project makes the workflow **retrieve evidence first, then answer** so responses stay tied to your data.

---

## What it does

- 📂 **Multi-format ingestion**: PDF, DOCX, TXT, CSV, JSON (and more depending on loaders)
- ✂️ **Chunking** with LangChain text splitters
- 🧠 **Embeddings** via Ollama embedding model (default: `nomic-embed-text`)
- 🔎 **Vector search** with `FAISS` (can be implemented with `ChromaDB` too)
- 🤖 **Local LLM generation** via Ollama (examples: Gemma, Mistral, Llama3)
- 🖥️ **Streamlit UI** for interactive document Q&A

---

## High-level architecture

1. **Load documents** from `/data`
2. **Split** into chunks (chunk size + overlap)
3. **Embed** chunks using an embedding model (Ollama embeddings)
4. **Store** embeddings in a FAISS index + metadata store
5. On query:
   - Embed the query
   - Retrieve top-k relevant chunks
   - Build a prompt with *ONLY retrieved context*
   - Generate an answer using a local Ollama LLM

---

## Tech stack

- **Python**
- **LangChain** (`langchain`, `langchain-community`, `langchain-text-splitters`)
- **Ollama** + `langchain-ollama` (local inference + embeddings)
- **FAISS** (vector index)
- **Streamlit** (UI)
- Loaders: `pypdf`, `docx2txt`, `unstructured` (and LangChain loaders)

---



## Prerequisites

### 1) Install Ollama
Install Ollama and make sure it’s running:

- Start server:
  ```bash
  ollama serve
  ```
### 2) Pull models (examples)

## Embedding model:
```
ollama pull nomic-embed-text
```

### LLM model (pick one):
```
ollama pull gemma3:1b
# or
ollama pull mistral
# or
ollama pull llama3
```

## Setup
```
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt
```

Note: faiss-cpu is commented in your requirements. If you need it:
```
pip install faiss-cpu
```
Add your data

Put your documents inside:

data/
  your_file.pdf
  notes.txt
  handbook.docx
  dataset.csv
  sample.json

## Run (Streamlit UI)
```
streamlit run streamlit_app.py
```

## Configuration

In streamlit_app.py, you can edit:

persist_dir: where FAISS index + metadata are stored

embedding_model: nomic-embed-text (or any Ollama embedding model)

llm_model: gemma3:1b, mistral, llama3, etc.

base_url: default Ollama URL http://localhost:11434

