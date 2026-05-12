"""
ollama_client.py
================
Thin wrapper around the local Ollama REST API.

Install Ollama:  https://ollama.com
Pull models:
    ollama pull nomic-embed-text
    ollama pull llama3.2
"""

from __future__ import annotations

import requests


class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434) -> None:
        self.base_url    = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*, or [] on failure."""
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        """Return the generated text for *prompt*, or an error string."""
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt,
                      "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"
