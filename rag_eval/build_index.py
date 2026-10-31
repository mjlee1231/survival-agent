"""Chunk the corpus and build the search index in data/index/.

    python rag_eval/build_index.py              # BM25 + dense (needs ".[rag]")
    python rag_eval/build_index.py --no-dense   # BM25 only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from survagent.rag.index import DEFAULT_MODEL, build_index

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chunk-size", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=40)
    args = ap.parse_args()
    m = build_index(ROOT / "data" / "corpus", ROOT / "data" / "index", dense=not args.no_dense,
                    model=args.model, size=args.chunk_size, overlap=args.overlap)
    print(f"Indexed {m['n_chunks']} passages; corpus sha256 {m['corpus_sha256'][:12]}"
          + (f"; dense model {m['dense_model']}" if "dense_model" in m else ""))


if __name__ == "__main__":
    main()
