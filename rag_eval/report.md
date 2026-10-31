# Retrieval evaluation

31 questions · top-10 retrieval · 2000 bootstrap resamples · seed 2026 · corpus 28ee1c5212ce (461 passages, 180-word chunks)

Means over questions with 95% percentile bootstrap intervals (questions resampled).

| Metric | bm25 | dense | hybrid |
| --- | --- | --- | --- |
| hit@1 | 0.19 (0.06–0.35) | 0.06 (0.00–0.16) | 0.19 (0.06–0.35) |
| hit@3 | 0.42 (0.26–0.61) | 0.32 (0.16–0.48) | 0.35 (0.19–0.52) |
| hit@5 | 0.45 (0.29–0.61) | 0.42 (0.26–0.58) | 0.45 (0.29–0.65) |
| hit@10 | 0.48 (0.32–0.68) | 0.65 (0.48–0.81) | 0.58 (0.42–0.74) |
| mrr@10 | 0.31 (0.18–0.46) | 0.23 (0.14–0.33) | 0.30 (0.18–0.44) |
| ndcg@10 | 0.29 (0.17–0.42) | 0.29 (0.20–0.38) | 0.30 (0.18–0.43) |

## Paired comparisons

Difference from the baseline on the same questions: paired bootstrap 95% interval and two-sided sign-flip randomization p-value (not adjusted for multiplicity).

| Comparison | Metric | Difference (95% CI) | p |
| --- | --- | --- | --- |
| dense − bm25 | hit@1 | -0.13 (-0.29 to +0.00) | 0.218 |
| dense − bm25 | hit@3 | -0.10 (-0.29 to +0.10) | 0.508 |
| dense − bm25 | hit@5 | -0.03 (-0.19 to +0.13) | 1.000 |
| dense − bm25 | hit@10 | +0.16 (+0.00 to +0.35) | 0.181 |
| dense − bm25 | mrr@10 | -0.08 (-0.18 to +0.04) | 0.205 |
| dense − bm25 | ndcg@10 | -0.01 (-0.09 to +0.09) | 0.912 |
| hybrid − bm25 | hit@1 | +0.00 (-0.10 to +0.10) | 1.000 |
| hybrid − bm25 | hit@3 | -0.06 (-0.19 to +0.06) | 0.621 |
| hybrid − bm25 | hit@5 | +0.00 (-0.10 to +0.10) | 1.000 |
| hybrid − bm25 | hit@10 | +0.10 (-0.03 to +0.23) | 0.371 |
| hybrid − bm25 | mrr@10 | -0.00 (-0.07 to +0.06) | 0.894 |
| hybrid − bm25 | ndcg@10 | +0.01 (-0.05 to +0.06) | 0.709 |

**Questions with no relevant passage in the hybrid top 10:** e9-03, e9-05, e9-06, e9-07, e9-09, e9-10, e9-11, e9-12, r1-01, r1-02, r1-03, r1-04, cv-04

## BM25 by chunk size

| Chunk size (words) | Passages | hit@5 | mrr@10 |
| --- | --- | --- | --- |
| 100 | 668 | 0.29 (0.13–0.45) | 0.23 |
| 180 | 466 | 0.45 (0.29–0.61) | 0.31 |
| 300 | 360 | 0.48 (0.32–0.65) | 0.33 |
