"""
main.py
=======
FastAPI server — Python port of the C++ VectorDB project.

Start:
    pip install -r requirements.txt
    python main.py
        or
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Then open http://localhost:8080 in your browser.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ollama_client import OllamaClient
from vector_store import (
    DIMS,
    DocumentDB,
    VectorDB,
    VectorItem,
    chunk_text,
    cosine,
    get_dist_fn,
)

# =====================================================================
#  APP SETUP
# =====================================================================

app = FastAPI(title="VectorDB — Python Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Global singletons
db     = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()


# =====================================================================
#  DEMO DATA  (20 × 16-D categorical vectors)
#  Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
# =====================================================================

def load_demo() -> None:
    dist = get_dist_fn("cosine")
    rows = [
        ("Linked List: nodes connected by pointers",               "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert",        "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems","cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal",   "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining",        "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits",             "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors",      "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem","math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography","math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions","math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes","food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls",          "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs","food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers",  "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks",   "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy",  "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves","sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game",  "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition","sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in rows:
        db.insert(meta, cat, emb, dist)


load_demo()

ollama_up = ollama.is_available()
print("=== VectorDB Engine (Python) ===")
print("http://localhost:8080")
print(f"{db.size()} demo vectors | {DIMS} dims | HNSW + KD-Tree + BruteForce")
print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
if ollama_up:
    print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")


# =====================================================================
#  REQUEST / RESPONSE MODELS
# =====================================================================

class InsertRequest(BaseModel):
    metadata:  str
    category:  str = ""
    embedding: list[float]


class DocInsertRequest(BaseModel):
    title: str
    text:  str


class DocSearchRequest(BaseModel):
    question: str
    k: int = 3


# =====================================================================
#  DEMO VECTOR ENDPOINTS
# =====================================================================

@app.get("/search")
def search(v: str, k: int = 5,
           metric: str = "cosine", algo: str = "hnsw"):
    q = _parse_vec(v)
    if len(q) != DIMS:
        raise HTTPException(400, detail=f"need {DIMS}D vector")
    out = db.search(q, k, metric, algo)
    return {
        "results": [
            {
                "id":        h.id,
                "metadata":  h.meta,
                "category":  h.cat,
                "distance":  round(h.dist, 6),
                "embedding": h.emb,
            }
            for h in out.hits
        ],
        "latencyUs": out.us,
        "algo":      out.algo,
        "metric":    out.metric,
    }


@app.post("/insert")
def insert(body: InsertRequest):
    if len(body.embedding) != DIMS:
        raise HTTPException(400, detail="invalid embedding size")
    id = db.insert(body.metadata, body.category,
                   body.embedding, get_dist_fn("cosine"))
    return {"id": id}


@app.delete("/delete/{id}")
def delete(id: int):
    ok = db.remove(id)
    return {"ok": ok}


@app.get("/items")
def items():
    return [
        {
            "id":        v.id,
            "metadata":  v.metadata,
            "category":  v.category,
            "embedding": v.emb,
        }
        for v in db.all()
    ]


@app.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    q = _parse_vec(v)
    if len(q) != DIMS:
        raise HTTPException(400, detail=f"need {DIMS}D vector")
    b = db.benchmark(q, k, metric)
    return {
        "bruteforceUs": b.bf_us,
        "kdtreeUs":     b.kd_us,
        "hnswUs":       b.hnsw_us,
        "itemCount":    b.n,
    }


@app.get("/hnsw-info")
def hnsw_info():
    return db.hnsw_info()


@app.get("/stats")
def stats():
    return {
        "count":      db.size(),
        "dims":       DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics":    ["euclidean", "cosine", "manhattan"],
    }


# =====================================================================
#  DOCUMENT + RAG ENDPOINTS
# =====================================================================

@app.post("/doc/insert")
def doc_insert(body: DocInsertRequest):
    if not body.title or not body.text:
        raise HTTPException(400, detail="need title and text")

    chunks = chunk_text(body.text, 250, 30)
    ids: list[int] = []

    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            raise HTTPException(503, detail=(
                "Ollama unavailable. Install from https://ollama.com "
                "then run: ollama pull nomic-embed-text && ollama pull llama3.2"
            ))
        chunk_title = (
            f"{body.title} [{i+1}/{len(chunks)}]"
            if len(chunks) > 1 else body.title
        )
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return {"ids": ids, "chunks": len(chunks), "dims": doc_db.dims}


@app.delete("/doc/delete/{id}")
def doc_delete(id: int):
    ok = doc_db.remove(id)
    return {"ok": ok}


@app.get("/doc/list")
def doc_list():
    return [
        {
            "id":      d.id,
            "title":   d.title,
            "preview": d.text[:120] + ("…" if len(d.text) > 120 else ""),
            "words":   len(d.text.split()),
        }
        for d in doc_db.all()
    ]


@app.post("/doc/search")
def doc_search(body: DocSearchRequest):
    if not body.question:
        raise HTTPException(400, detail="need question")
    q_emb = ollama.embed(body.question)
    if not q_emb:
        raise HTTPException(503, detail="Ollama unavailable")
    hits = doc_db.search(q_emb, body.k)
    return {
        "contexts": [
            {"id": item.id, "title": item.title,
             "distance": round(d, 4)}
            for d, item in hits
        ]
    }


@app.post("/doc/ask")
def doc_ask(body: DocSearchRequest):
    if not body.question:
        raise HTTPException(400, detail="need question")

    # Step 1: embed the question
    q_emb = ollama.embed(body.question)
    if not q_emb:
        raise HTTPException(503, detail="Ollama unavailable")

    # Step 2: retrieve top-k relevant chunks
    hits = doc_db.search(q_emb, body.k)

    # Step 3: build prompt
    ctx_parts = [
        f"[{i+1}] {item.title}:\n{item.text}\n"
        for i, (_, item) in enumerate(hits)
    ]
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say "
        "things like 'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        "Context:\n" + "\n".join(ctx_parts) +
        f"\nQuestion: {body.question}\n\nAnswer:"
    )

    # Step 4: generate answer
    answer = ollama.generate(prompt)

    # Step 5: return everything
    return {
        "answer":   answer,
        "model":    ollama.gen_model,
        "contexts": [
            {
                "id":       item.id,
                "title":    item.title,
                "text":     item.text,
                "distance": round(d, 4),
            }
            for d, item in hits
        ],
        "docCount": doc_db.size(),
    }


# =====================================================================
#  STATUS + FRONTEND
# =====================================================================

@app.get("/status")
def status():
    up = ollama.is_available()
    return {
        "ollamaAvailable": up,
        "embedModel":      ollama.embed_model,
        "genModel":        ollama.gen_model,
        "docCount":        doc_db.size(),
        "docDims":         doc_db.dims,
        "demoDims":        DIMS,
        "demoCount":       db.size(),
    }


@app.get("/")
def index():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(404, detail="index.html not found")
    return FileResponse(str(html_path), media_type="text/html")


# =====================================================================
#  HELPERS
# =====================================================================

def _parse_vec(s: str) -> list[float]:
    """Parse a comma-separated float string into a list."""
    result = []
    for token in s.split(","):
        token = token.strip()
        try:
            result.append(float(token))
        except ValueError:
            pass
    return result


# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
