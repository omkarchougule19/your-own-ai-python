# VectorDB — Build a Vector Database from Scratch in Python

A fully working **vector database built from scratch in Python**, with a web UI.
Implements **HNSW**, **KD-Tree**, and **Brute Force** k-NN search in pure Python,
plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to show how production vector databases like
> Pinecone, Weaviate, and Chroma actually work under the hood.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16-D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2-D PCA Scatter Plot** | Live visualization of the semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768-D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)     ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)           ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search               ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)             ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by
Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each
layer is progressively sparser — searches start at the top layer and zoom in,
achieving O(log N) complexity instead of O(N) for brute force.

---

## Prerequisites

You need **2 things**:

1. **Python 3.10+**
2. **Ollama** — runs the local AI models *(optional, only needed for document RAG)*

---

## Quick Start

### 1 — Clone and install dependencies

```bash
git clone <your-repo-url>
cd your-own-ai-python
pip install -r requirements.txt
```

### 2 — (Optional) Set up Ollama

The demo vector database works out of the box. You only need Ollama if you want
to insert real documents and ask questions about them.

```bash
# Download from https://ollama.com, then:
ollama pull nomic-embed-text
ollama pull llama3.2
ollama serve          # starts the local API on localhost:11434
```

### 3 — Start the server

```bash
python main.py
```

Open **http://localhost:8080** in your browser.

For development with auto-reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

---

## Project Layout

```
your-own-ai-python/
├── main.py            # FastAPI server + all REST endpoints + demo data loader
├── vector_store.py    # BruteForce, KDTree, HNSW, VectorDB, DocumentDB, chunk_text
├── ollama_client.py   # Thin wrapper around the Ollama REST API
├── index.html         # Web UI (served at /)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## REST API Reference

### Demo Vector Endpoints

| Method | Path | Params / Body | Description |
|--------|------|---------------|-------------|
| `GET` | `/search` | `v`, `k`, `metric`, `algo` | k-NN search |
| `POST` | `/insert` | `{metadata, category, embedding}` | Insert a vector |
| `DELETE` | `/delete/{id}` | — | Delete a vector |
| `GET` | `/items` | — | List all vectors |
| `GET` | `/benchmark` | `v`, `k`, `metric` | Benchmark all 3 algorithms |
| `GET` | `/hnsw-info` | — | HNSW graph structure |
| `GET` | `/stats` | — | Item count, dims, supported algorithms and metrics |

**Search params:**
- `v` — comma-separated floats, must be 16-D for the demo index
- `k` — number of results (default `5`)
- `metric` — `cosine` (default) | `euclidean` | `manhattan`
- `algo` — `hnsw` (default) | `kdtree` | `bruteforce`

### Document / RAG Endpoints

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/doc/insert` | `{title, text}` | Chunk + embed a document via Ollama |
| `DELETE` | `/doc/delete/{id}` | — | Delete a chunk |
| `GET` | `/doc/list` | — | List all stored chunks |
| `POST` | `/doc/search` | `{question, k}` | Semantic retrieval only |
| `POST` | `/doc/ask` | `{question, k}` | Full RAG: embed → retrieve → generate |
| `GET` | `/status` | — | Ollama availability + DB sizes |

---

## Architecture Notes

### Pure-Python algorithms

All three search algorithms are implemented from scratch in `vector_store.py`
with no external dependencies beyond the standard library.

**BruteForce** computes the distance from the query to every stored vector and
returns the k smallest — O(N) per query, always exact.

**KD-Tree** recursively partitions the vector space by alternating dimensions.
Queries prune branches using the splitting-plane distance — much faster than
brute force in low dimensions, but degrades as dimensionality grows.

**HNSW** builds a layered proximity graph. The top layers are sparse long-range
links used for fast navigation; the bottom layer is a dense graph used for
precise local search. Insert and query are both O(log N) on average.

### Thread safety

Every public method on `VectorDB` and `DocumentDB` holds a `threading.Lock`
for the duration of the operation. FastAPI runs handlers in a thread pool,
so concurrent requests are safe.

### Ollama integration

`ollama_client.py` wraps two Ollama endpoints:

- `POST /api/embeddings` — produces a float vector for a text input
- `POST /api/generate` — produces a text completion given a prompt

Both calls use `requests` with explicit timeouts (30 s for embeddings, 180 s
for generation). If Ollama is not running, embedding calls return `[]` and
the server responds with a `503`.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for package versions

```
fastapi
uvicorn[standard]
requests
pydantic
```
