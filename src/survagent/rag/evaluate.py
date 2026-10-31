"""Statistical evaluation of retrieval.

Relevance labels are *evidence phrases*: a passage is relevant to a question if it
comes from the named document and contains one of the phrases. Labels therefore
do not depend on how documents are chunked, so different chunkings and
retrievers can be compared on the same questions.

Uncertainty: questions are the sampling unit. Metric means get percentile
bootstrap CIs; differences between methods are paired (same questions) with a
paired bootstrap CI and a sign-flip randomization test.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from .corpus import Chunk


def _norm(text: str) -> str:
    text = text.lower().replace("‐", "-").replace("‑", "-").replace("–", "-")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text)


def load_questions(path: str | Path) -> list[dict]:
    qs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        raise ValueError("Question ids must be unique.")
    return qs


def is_relevant(chunk: Chunk, question: dict) -> bool:
    text = _norm(chunk.text)
    return any(
        chunk.doc_id == rel["doc"] and any(_norm(p) in text for p in rel["phrases"])
        for rel in question["relevant"]
    )


def validate_labels(chunks: Sequence[Chunk], questions: Sequence[dict]) -> list[dict]:
    """Count relevant chunks per question. A question with none cannot be
    evaluated (its phrases are not in the corpus) and must be fixed."""
    out = []
    for q in questions:
        n = sum(is_relevant(c, q) for c in chunks)
        out.append({"id": q["id"], "n_relevant": n, "ok": n > 0})
    return out


def relevance_matrix(index, questions: Sequence[dict], method: str, depth: int = 10) -> np.ndarray:
    """Binary matrix (questions x ranks): 1 where the passage at that rank is relevant."""
    R = np.zeros((len(questions), depth))
    for qi, q in enumerate(questions):
        hits = index.retrievers[method].search(q["question"], depth)
        for r, (ci, _) in enumerate(hits):
            R[qi, r] = is_relevant(index.chunks[ci], q)
    return R


def per_question_metrics(R: np.ndarray, n_relevant: np.ndarray) -> dict[str, np.ndarray]:
    depth = R.shape[1]
    first = np.where(R.any(axis=1), R.argmax(axis=1) + 1, 0)
    rr = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
    discounts = 1 / np.log2(np.arange(2, depth + 2))
    dcg = R @ discounts
    ideal = np.array([discounts[: min(int(n), depth)].sum() for n in n_relevant])
    out = {f"hit@{k}": R[:, :k].any(axis=1).astype(float) for k in (1, 3, 5, 10) if k <= depth}
    out[f"mrr@{depth}"] = rr
    out[f"ndcg@{depth}"] = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
    return out


def bootstrap_mean_ci(x: np.ndarray, rng: np.random.Generator, B: int = 2000, level: float = 0.95):
    idx = rng.integers(0, x.size, size=(B, x.size))
    means = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def sign_flip_pvalue(d: np.ndarray, rng: np.random.Generator, n_perm: int = 10000) -> float:
    """Two-sided paired randomization test of mean difference = 0."""
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1))


def evaluate(index, questions: Sequence[dict], methods: Sequence[str], depth: int = 10,
             B: int = 2000, seed: int = 2026, baseline: str | None = None) -> dict:
    labels = validate_labels(index.chunks, questions)
    bad = [l["id"] for l in labels if not l["ok"]]
    if bad:
        raise ValueError(f"No relevant passage in the corpus for questions {bad}; fix their phrases.")
    n_rel = np.array([l["n_relevant"] for l in labels])
    rng = np.random.default_rng(seed)

    per_q: dict[str, dict[str, np.ndarray]] = {}
    summary = {}
    for m in methods:
        R = relevance_matrix(index, questions, m, depth)
        per_q[m] = per_question_metrics(R, n_rel)
        summary[m] = {
            metric: {"mean": float(v.mean()), "ci_95": bootstrap_mean_ci(v, rng, B)}
            for metric, v in per_q[m].items()
        }

    comparisons = []
    baseline = baseline or methods[0]
    for m in methods:
        if m == baseline:
            continue
        for metric in per_q[m]:
            d = per_q[m][metric] - per_q[baseline][metric]
            comparisons.append({
                "method": m, "baseline": baseline, "metric": metric,
                "mean_difference": float(d.mean()),
                "ci_95": bootstrap_mean_ci(d, rng, B),
                "p_value": sign_flip_pvalue(d, rng),
            })

    first_method = methods[-1]
    misses = [q["id"] for q, h in zip(questions, per_q[first_method][f"hit@{depth}"]) if h == 0]
    return {
        "n_questions": len(questions),
        "depth": depth,
        "bootstrap_reps": B,
        "seed": seed,
        "summary": summary,
        "comparisons": comparisons,
        "misses": {first_method: misses},
        "per_question": {m: {k: v.tolist() for k, v in pq.items()} for m, pq in per_q.items()},
        "question_ids": [q["id"] for q in questions],
    }


def render_report(result: dict, manifest: dict) -> str:
    depth = result["depth"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"{result['n_questions']} questions · top-{depth} retrieval · {result['bootstrap_reps']} "
        f"bootstrap resamples · seed {result['seed']} · corpus {str(manifest.get('corpus_sha256'))[:12]} "
        f"({manifest.get('n_chunks')} passages, {manifest.get('chunk_size')}-word chunks)",
        "",
        "Means over questions with 95% percentile bootstrap intervals (questions resampled).",
        "",
    ]
    methods = list(result["summary"])
    metrics = list(result["summary"][methods[0]])
    lines.append("| Metric | " + " | ".join(methods) + " |")
    lines.append("| --- |" + " --- |" * len(methods))
    for metric in metrics:
        cells = []
        for m in methods:
            s = result["summary"][m][metric]
            cells.append(f"{s['mean']:.2f} ({s['ci_95'][0]:.2f}–{s['ci_95'][1]:.2f})")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    if result["comparisons"]:
        lines += [
            "",
            "## Paired comparisons",
            "",
            "Difference from the baseline on the same questions: paired bootstrap 95% interval and "
            "two-sided sign-flip randomization p-value (not adjusted for multiplicity).",
            "",
            "| Comparison | Metric | Difference (95% CI) | p |",
            "| --- | --- | --- | --- |",
        ]
        for c in result["comparisons"]:
            lines.append(
                f"| {c['method']} − {c['baseline']} | {c['metric']} | {c['mean_difference']:+.2f} "
                f"({c['ci_95'][0]:+.2f} to {c['ci_95'][1]:+.2f}) | {c['p_value']:.3f} |"
            )
    for m, ids in result["misses"].items():
        lines += ["", f"**Questions with no relevant passage in the {m} top {depth}:** "
                  + (", ".join(ids) if ids else "none")]
    lines.append("")
    return "\n".join(lines)
