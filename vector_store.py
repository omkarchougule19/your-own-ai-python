"""
vector_store.py
===============
Pure-Python implementations of:
  - Distance metrics (Euclidean, Cosine, Manhattan)
  - BruteForce k-NN
  - KD-Tree k-NN
  - HNSW (Hierarchical Navigable Small World) k-NN
  - VectorDB  – demo 16-D index combining all three
  - DocumentDB – HNSW over real Ollama embeddings (for RAG)
  - chunk_text utility
"""

from __future__ import annotations

import heapq
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16  # dimensionality of demo vectors

# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: list[float]


DistFn = Callable[[list[float], list[float]], float]


# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a)
    nb  = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self) -> None:
        self.items: list[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        results = sorted((dist(q, v.emb), v.id) for v in self.items)
        return results[:k]

    def remove(self, id: int) -> None:
        self.items = [v for v in self.items if v.id != id]


# =====================================================================
#  KD-TREE
# =====================================================================

class _KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem) -> None:
        self.item  = item
        self.left: Optional[_KDNode]  = None
        self.right: Optional[_KDNode] = None


class KDTree:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self._root: Optional[_KDNode] = None

    # ── public API ────────────────────────────────────────────────────

    def insert(self, v: VectorItem) -> None:
        self._root = self._insert(self._root, v, 0)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        heap: list[tuple[float, int]] = []   # max-heap via negated distance
        self._knn(self._root, q, k, 0, dist, heap)
        return sorted((-d, id) for d, id in heap)

    def rebuild(self, items: list[VectorItem]) -> None:
        self._root = None
        for v in items:
            self.insert(v)

    # ── internal helpers ──────────────────────────────────────────────

    def _insert(self, node: Optional[_KDNode], v: VectorItem,
                depth: int) -> _KDNode:
        if node is None:
            return _KDNode(v)
        ax = depth % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left  = self._insert(node.left,  v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def _knn(self, node: Optional[_KDNode], q: list[float], k: int,
             depth: int, dist: DistFn,
             heap: list[tuple[float, int]]) -> None:
        if node is None:
            return

        dn = dist(q, node.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)

        ax   = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer,  q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)


# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class _HNSWNode:
    __slots__ = ("item", "max_lyr", "nbrs")

    def __init__(self, item: VectorItem, max_lyr: int) -> None:
        self.item    = item
        self.max_lyr = max_lyr
        self.nbrs: list[list[int]] = [[] for _ in range(max_lyr + 1)]


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200) -> None:
        self.M        = m
        self.M0       = 2 * m
        self.ef_build = ef_build
        self.mL       = 1.0 / math.log(m)
        self.G: dict[int, _HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt  = -1
        self._rng      = random.Random(42)

    # ── public API ────────────────────────────────────────────────────

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        id  = item.id
        lvl = self._rand_level()
        self.G[id] = _HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt  = id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            ep_node = self.G.get(ep)
            if ep_node and lc < len(ep_node.nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W      = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m  = self.M0 if lc == 0 else self.M
            sel    = self._select_nbrs(W, max_m)
            self.G[id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                nd = self.G[nid]
                while lc >= len(nd.nbrs):
                    nd.nbrs.append([])
                conn = nd.nbrs[lc]
                conn.append(id)
                if len(conn) > max_m:
                    ds = [(dist(nd.item.emb, self.G[c].item.emb), c)
                          for c in conn if c in self.G]
                    ds.sort()
                    nd.nbrs[lc] = [c for _, c in ds[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt  = id

    def knn(self, q: list[float], k: int, ef: int,
            dist: DistFn) -> list[tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            ep_node = self.G.get(ep)
            if ep_node and lc < len(ep_node.nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id: int) -> None:
        if id not in self.G:
            return
        for nd in self.G.values():
            for layer in nd.nbrs:
                try:
                    layer.remove(id)
                except ValueError:
                    pass
        if self.entry_pt == id:
            self.entry_pt = next((nid for nid in self.G if nid != id), -1)
        del self.G[id]

    def get_info(self) -> dict:
        max_l           = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes: list[dict] = []
        edges: list[dict] = []

        for id, nd in self.G.items():
            nodes.append({
                "id":       id,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr":   nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id, "dst": nid, "lyr": lc})

        return {
            "topLayer":       self.top_layer,
            "nodeCount":      len(self.G),
            "nodesPerLayer":  nodes_per_layer,
            "edgesPerLayer":  edges_per_layer,
            "nodes":          nodes,
            "edges":          edges,
        }

    def size(self) -> int:
        return len(self.G)

    # ── internal helpers ──────────────────────────────────────────────

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: list[float], ep: int, ef: int, lyr: int,
                       dist: DistFn) -> list[tuple[float, int]]:
        vis: set[int] = {ep}
        d0    = dist(q, self.G[ep].item.emb)
        cands = [(d0, ep)]          # min-heap
        found = [(-d0, ep)]         # max-heap (negated)

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted((-d, id) for d, id in found)

    @staticmethod
    def _select_nbrs(cands: list[tuple[float, int]],
                     max_m: int) -> list[int]:
        return [id for _, id in cands[:max_m]]


# =====================================================================
#  VECTOR DATABASE  (demo 16-D index)
# =====================================================================

@dataclass
class SearchHit:
    id:   int
    meta: str
    cat:  str
    emb:  list[float]
    dist: float


@dataclass
class SearchOut:
    hits:   list[SearchHit]
    us:     int   # latency in microseconds
    algo:   str
    metric: str


@dataclass
class BenchOut:
    bf_us:   int
    kd_us:   int
    hnsw_us: int
    n:       int


class VectorDB:
    def __init__(self, dims: int) -> None:
        self.dims     = dims
        self._store:  dict[int, VectorItem] = {}
        self._bf      = BruteForce()
        self._kdt     = KDTree(dims)
        self._hnsw    = HNSW(16, 200)
        self._lock    = threading.Lock()
        self._next_id = 1

    def insert(self, meta: str, cat: str, emb: list[float],
               dist: DistFn) -> int:
        with self._lock:
            v = VectorItem(id=self._next_id, metadata=meta,
                           category=cat, emb=emb)
            self._next_id += 1
            self._store[v.id] = v
            self._bf.insert(v)
            self._kdt.insert(v)
            self._hnsw.insert(v, dist)
            return v.id

    def remove(self, id: int) -> bool:
        with self._lock:
            if id not in self._store:
                return False
            del self._store[id]
            self._bf.remove(id)
            self._hnsw.remove(id)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, q: list[float], k: int,
               metric: str, algo: str) -> SearchOut:
        with self._lock:
            dfn = get_dist_fn(metric)
            t0  = time.perf_counter()

            if algo == "bruteforce":
                raw = self._bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self._kdt.knn(q, k, dfn)
            else:
                raw = self._hnsw.knn(q, k, 50, dfn)

            us   = int((time.perf_counter() - t0) * 1_000_000)
            hits = [
                SearchHit(id=id, meta=self._store[id].metadata,
                          cat=self._store[id].category,
                          emb=self._store[id].emb, dist=d)
                for d, id in raw if id in self._store
            ]
            return SearchOut(hits=hits, us=us, algo=algo, metric=metric)

    def benchmark(self, q: list[float], k: int, metric: str) -> BenchOut:
        with self._lock:
            dfn = get_dist_fn(metric)

            def timed(fn) -> int:
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            return BenchOut(
                bf_us   = timed(lambda: self._bf.knn(q, k, dfn)),
                kd_us   = timed(lambda: self._kdt.knn(q, k, dfn)),
                hnsw_us = timed(lambda: self._hnsw.knn(q, k, 50, dfn)),
                n       = len(self._store),
            )

    def all(self) -> list[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self._store)


# =====================================================================
#  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
# =====================================================================

@dataclass
class DocItem:
    id:    int
    title: str
    text:  str
    emb:   list[float]


class DocumentDB:
    def __init__(self) -> None:
        self._store:  dict[int, DocItem] = {}
        self._hnsw    = HNSW(16, 200)
        self._bf      = BruteForce()
        self._lock    = threading.Lock()
        self._next_id = 1
        self.dims     = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self._lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(id=self._next_id, title=title,
                           text=text, emb=emb)
            self._next_id += 1
            self._store[item.id] = item
            vi = VectorItem(id=item.id, metadata=title,
                            category="doc", emb=emb)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, q: list[float], k: int,
               max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < 10:
                raw = self._bf.knn(q, k, cosine)
            else:
                raw = self._hnsw.knn(q, k, 50, cosine)
            return [
                (d, self._store[id])
                for d, id in raw
                if id in self._store and d <= max_dist
            ]

    def remove(self, id: int) -> bool:
        with self._lock:
            if id not in self._store:
                return False
            del self._store[id]
            self._hnsw.remove(id)
            self._bf.remove(id)
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)


# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250,
               overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks: list[str] = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks
