"""Lexical (BM25), dense (embedding) and hybrid retrieval over chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Protocol, Sequence

import numpy as np

from .corpus import Chunk

_STOP = set(
    "a an and are as at be been but by can could did do does for from had has have how if in "
    "into is it its may might more most must no not of on or should such than that the their "
    "then there these they this those to under was we were what when where which while who "
    "will with would you your".split()
)


def _stem(tok: str) -> str:
    # Deliberately light: fold common plurals so "estimands" matches "estimand".
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in toks if t not in _STOP and (len(t) > 1 or t.isdigit())]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[tuple[int, float]]: ...


class BM25:
    """Okapi BM25 (Robertson & Zaragoza, 2009)."""

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [tokenize(c.title + " " + c.text) for c in chunks]
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n)
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, ()):
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s[i] += self.idf[term] * tf * (self.k1 + 1) / norm
        return s

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        return _top_k(self.scores(query), k, positive_only=True)  # no term overlap = not retrieved


Embedder = Callable[[list[str]], np.ndarray]


def sentence_transformer_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """Embedding function backed by sentence-transformers (``pip install -e ".[rag]"``)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            'Dense retrieval needs sentence-transformers: pip install -e ".[rag]"'
        ) from exc
    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, batch_size=32, normalize_embeddings=True,
                                       show_progress_bar=False), dtype=np.float32)

    embed.model_name = model_name  # type: ignore[attr-defined]
    return embed


class Dense:
    """Cosine similarity between query and chunk embeddings."""

    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], embed: Embedder, vectors: np.ndarray | None = None):
        self.embed = embed
        self.vectors = vectors if vectors is not None else embed([c.title + ". " + c.text for c in chunks])
        if len(self.vectors) != len(chunks):
            raise ValueError("Stored embeddings do not match the corpus; rebuild the index.")

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = self.embed([query])[0]
        return _top_k(self.vectors @ q, k)


class Hybrid:
    """Reciprocal rank fusion (Cormack et al., 2009): score = sum over retrievers
    of 1 / (c + rank). Uses ranks only, so BM25 and cosine scores need no rescaling."""

    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], c: int = 60, depth: int = 100):
        self.retrievers, self.c, self.depth = list(retrievers), c, depth

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for r in self.retrievers:
            for rank, (i, _) in enumerate(r.search(query, self.depth), start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.c + rank)
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def _top_k(scores: np.ndarray, k: int, positive_only: bool = False) -> list[tuple[int, float]]:
    """Indices of the k highest scores, ties broken by position (deterministic)."""
    k = min(k, scores.size)
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.lexsort((idx, -scores[idx]))]
    return [(int(i), float(scores[i])) for i in idx if not positive_only or scores[i] > 0]
