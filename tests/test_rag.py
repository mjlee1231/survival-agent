"""Retrieval, labelling and evaluation logic on a small synthetic corpus."""

import json

import numpy as np
import pytest

from survagent.rag.corpus import Chunk, chunk_pages, chunk_words, load_corpus
from survagent.rag.evaluate import (
    evaluate,
    is_relevant,
    per_question_metrics,
    sign_flip_pvalue,
    validate_labels,
)
from survagent.rag.index import GuidanceIndex
from survagent.rag.retrievers import BM25, Hybrid, tokenize

DOCS = {
    "a": "The intention-to-treat principle means all randomized subjects are analysed in the "
         "groups to which they were assigned. The full analysis set is close to this ideal.",
    "b": "Interim analyses require control of the type I error, for example with an alpha "
         "spending function chosen in advance.",
    "c": "A treatment policy strategy uses the outcome regardless of intercurrent events such "
         "as use of rescue medication.",
    "d": "Covariate adjustment for prognostic baseline variables can increase power; the "
         "covariates should be prespecified in the protocol.",
}
CHUNKS = [Chunk(f"{k}:p1:0", k, f"Doc {k}", f"https://example.org/{k}", 1, v) for k, v in DOCS.items()]
QUESTIONS = [
    {"id": "q1", "question": "who is in the full analysis set", "relevant": [{"doc": "a", "phrases": ["full analysis set"]}]},
    {"id": "q2", "question": "alpha spending at interim analyses", "relevant": [{"doc": "b", "phrases": ["alpha spending"]}]},
    {"id": "q3", "question": "rescue medication treatment policy", "relevant": [{"doc": "c", "phrases": ["treatment policy"]}]},
    {"id": "q4", "question": "prespecified covariate adjustment", "relevant": [{"doc": "d", "phrases": ["prespecified"]}]},
]


def hashing_embedder(texts):
    """Deterministic stand-in for a sentence-embedding model (bag of hashed tokens)."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in tokenize(t):
            out[i, hash(tok) % 256] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def test_tokenize_folds_plurals_and_drops_stopwords():
    assert tokenize("The estimands of analyses") == ["estimand", "analyse"]


def test_chunk_words_overlap_and_coverage():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_words(" ".join(words), size=10, overlap=3)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert set(" ".join(chunks).split()) == set(words)


def test_chunk_pages_keeps_page_numbers():
    chunks = chunk_pages("doc", "T", "u", ["page one text", "page two text"])
    assert [c.page for c in chunks] == [1, 2]


def test_bm25_ranks_matching_document_first():
    bm = BM25(CHUNKS)
    for q, doc in zip(QUESTIONS, "abcd"):
        assert CHUNKS[bm.search(q["question"], 1)[0][0]].doc_id == doc


def test_bm25_returns_nothing_without_term_overlap():
    assert BM25(CHUNKS).search("zebra giraffe", 3) == []


def test_reciprocal_rank_fusion_scores():
    class Fixed:
        def __init__(self, order):
            self.order = order

        def search(self, q, k):
            return [(i, 1.0) for i in self.order[:k]]

    fused = Hybrid([Fixed([0, 1, 2]), Fixed([2, 0, 1])], c=60).search("x", 3)
    assert fused[0][0] == 0  # ranks 1 and 2 beat ranks 3 and 1
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_relevance_requires_document_and_phrase():
    q = QUESTIONS[0]
    assert is_relevant(CHUNKS[0], q)
    other = Chunk("x", "b", "", "", 1, "the full analysis set")
    assert not is_relevant(other, q)  # right phrase, wrong document


def test_validate_labels_flags_missing_evidence():
    q = {"id": "bad", "question": "?", "relevant": [{"doc": "a", "phrases": ["not in corpus"]}]}
    assert validate_labels(CHUNKS, [q])[0]["ok"] is False


def test_metrics_on_known_rankings():
    R = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    m = per_question_metrics(R, np.array([1, 1, 1]))
    assert m["hit@1"].tolist() == [1, 0, 0]
    assert m["mrr@3"].tolist() == pytest.approx([1, 1 / 3, 0])
    assert m["ndcg@3"].tolist() == pytest.approx([1, 0.5, 0])


def test_sign_flip_pvalue_extremes():
    rng = np.random.default_rng(0)
    assert sign_flip_pvalue(np.zeros(10), rng) == 1.0
    assert sign_flip_pvalue(np.ones(20), rng) < 0.001


def test_evaluate_end_to_end_with_dense_and_hybrid():
    index = GuidanceIndex.from_chunks(CHUNKS, embed=hashing_embedder)
    res = evaluate(index, QUESTIONS, ["bm25", "dense", "hybrid"], depth=3, B=200)
    assert res["summary"]["bm25"]["hit@1"]["mean"] == 1.0
    assert {c["method"] for c in res["comparisons"]} == {"dense", "hybrid"}
    lo, hi = res["summary"]["dense"]["mrr@3"]["ci_95"]
    assert 0 <= lo <= hi <= 1


def test_evaluate_refuses_unlabelled_questions():
    q = {"id": "bad", "question": "x", "relevant": [{"doc": "a", "phrases": ["absent phrase"]}]}
    with pytest.raises(ValueError, match="fix their phrases"):
        evaluate(GuidanceIndex.from_chunks(CHUNKS), [q], ["bm25"], depth=3, B=50)


def test_search_output_has_citations():
    out = GuidanceIndex.from_chunks(CHUNKS).search("alpha spending", k=2)
    top = out["results"][0]
    assert top["chunk_id"] == "b:p1:0" and top["page"] == 1 and top["source"].startswith("https://")


def test_load_corpus_reads_pubmed_and_hashes(tmp_path):
    (tmp_path / "pubmed.jsonl").write_text(json.dumps(
        {"pmid": "1", "title": "RMST", "abstract": "Restricted mean survival time.", "journal": "J", "year": "2020"}
    ) + "\n")
    (tmp_path / "sources.json").write_text(json.dumps([{"kind": "pubmed", "file": "pubmed.jsonl"}]))
    chunks, manifest = load_corpus(tmp_path)
    assert chunks[0].chunk_id == "pmid:1" and chunks[0].source.startswith("PMID 1")
    assert len(manifest["corpus_sha256"]) == 64
