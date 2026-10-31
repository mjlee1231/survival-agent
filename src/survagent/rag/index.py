"""Build, save and load the search index; the search used by agents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .corpus import Chunk, load_corpus
from .retrievers import BM25, Dense, Hybrid, sentence_transformer_embedder

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(corpus_dir: str | Path, index_dir: str | Path, dense: bool = True,
                model: str = DEFAULT_MODEL, size: int = 180, overlap: int = 40) -> dict:
    """Chunk the corpus, embed it (optionally) and write the index to ``index_dir``."""
    chunks, manifest = load_corpus(corpus_dir, size, overlap)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    if dense:
        embed = sentence_transformer_embedder(model)
        vectors = Dense(chunks, embed).vectors
        np.save(index_dir / "dense.npy", vectors)
        manifest["dense_model"] = model
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class GuidanceIndex:
    """Loaded index with BM25, dense (if built) and hybrid search."""

    def __init__(self, index_dir: str | Path):
        index_dir = Path(index_dir)
        if not (index_dir / "manifest.json").is_file():
            raise ValueError(
                f"No index in {index_dir}. Run `python rag_eval/fetch_corpus.py` and "
                "`python rag_eval/build_index.py` first."
            )
        self.manifest = json.loads((index_dir / "manifest.json").read_text())
        self.chunks = [Chunk(**json.loads(line)) for line in open(index_dir / "chunks.jsonl")]
        self.retrievers = {"bm25": BM25(self.chunks)}
        if "dense_model" in self.manifest:
            embed = sentence_transformer_embedder(self.manifest["dense_model"])
            self.retrievers["dense"] = Dense(self.chunks, embed, np.load(index_dir / "dense.npy"))
            self.retrievers["hybrid"] = Hybrid([self.retrievers["bm25"], self.retrievers["dense"]])

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], embed=None) -> "GuidanceIndex":
        """In-memory index (used in tests and evaluation experiments)."""
        self = cls.__new__(cls)
        self.manifest = {"n_chunks": len(chunks), "corpus_sha256": None}
        self.chunks = list(chunks)
        self.retrievers = {"bm25": BM25(self.chunks)}
        if embed is not None:
            self.retrievers["dense"] = Dense(self.chunks, embed)
            self.retrievers["hybrid"] = Hybrid([self.retrievers["bm25"], self.retrievers["dense"]])
        return self

    def search(self, query: str, k: int = 5, method: str | None = None) -> dict:
        if not query.strip():
            raise ValueError("`query` is empty.")
        if not 1 <= k <= 50:
            raise ValueError("`k` must be between 1 and 50.")
        method = method or ("hybrid" if "hybrid" in self.retrievers else "bm25")
        if method not in self.retrievers:
            raise ValueError(f"`method` must be one of {sorted(self.retrievers)} for this index.")
        hits = self.retrievers[method].search(query, k)
        return {
            "method": method,
            "query": query,
            "results": [
                {
                    "rank": r,
                    "score": round(s, 4),
                    "chunk_id": self.chunks[i].chunk_id,
                    "title": self.chunks[i].title,
                    "page": self.chunks[i].page,
                    "source": self.chunks[i].source,
                    "text": self.chunks[i].text,
                }
                for r, (i, s) in enumerate(hits, start=1)
            ],
            "corpus_sha256": self.manifest.get("corpus_sha256"),
            "notes": [] if hits else ["No passage matched; rephrase with terms used in guidance documents."],
        }
