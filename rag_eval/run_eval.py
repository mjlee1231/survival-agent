"""Evaluate retrieval methods on the labelled question set.

    python rag_eval/run_eval.py --check-labels   # every question has evidence in the corpus?
    python rag_eval/run_eval.py                  # full evaluation -> rag_eval/report.md

Also compares chunk sizes for BM25 (--chunk-sizes 100 180 300), which is
possible because labels are phrases, not chunk ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from survagent.rag.corpus import load_corpus
from survagent.rag.evaluate import evaluate, load_questions, render_report, validate_labels
from survagent.rag.index import GuidanceIndex

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-labels", action="store_true")
    ap.add_argument("--chunk-sizes", type=int, nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    questions = load_questions(HERE / "questions.jsonl")
    index = GuidanceIndex(ROOT / "data" / "index")

    if args.check_labels:
        bad = 0
        for row in validate_labels(index.chunks, questions):
            flag = "ok " if row["ok"] else "MISSING"
            bad += not row["ok"]
            print(f"{flag} {row['id']:<8} {row['n_relevant']:3d} relevant passages")
        print(f"\n{len(questions) - bad}/{len(questions)} questions have evidence in the corpus.")
        return

    methods = [m for m in ("bm25", "dense", "hybrid") if m in index.retrievers]
    result = evaluate(index, questions, methods, B=args.reps, seed=args.seed, baseline="bm25")
    report = render_report(result, index.manifest)

    if args.chunk_sizes:
        report += "\n## BM25 by chunk size\n\n| Chunk size (words) | Passages | hit@5 | mrr@10 |\n| --- | --- | --- | --- |\n"
        for size in args.chunk_sizes:
            chunks, manifest = load_corpus(ROOT / "data" / "corpus", size=size, overlap=size // 4)
            r = evaluate(GuidanceIndex.from_chunks(chunks), questions, ["bm25"], B=args.reps, seed=args.seed)
            s = r["summary"]["bm25"]
            report += (f"| {size} | {len(chunks)} | {s['hit@5']['mean']:.2f} ({s['hit@5']['ci_95'][0]:.2f}–"
                       f"{s['hit@5']['ci_95'][1]:.2f}) | {s['mrr@10']['mean']:.2f} |\n")

    (HERE / "report.md").write_text(report)
    (HERE / "results.json").write_text(json.dumps(result, indent=2))
    print(report)


if __name__ == "__main__":
    main()
