# Step 39 — Learning-to-Rank with Cross-Strategy Consensus

> **Status: tried, not adopted — implementation removed.** Code, tests, and the regenerable parquet/CSV/JSON artefacts have been deleted. This document is the experiment record. Numbers below come from a single end-to-end LOO-CV run on 2026-05-07 against `data/dataset/dataset.parquet` (45 programmes, 520 jobs).


## Setup

- **Candidate pool:** per-programme top-50 by semantic cosine (matches Stage 1 of `align_hybrid`)
- **Label source:** consensus over `{symbolic, semantic, BM25}`, `≥ 2` of 3 in top-10
  - Hybrid is intentionally excluded to avoid training the learner against the formula it is meant to replace.
- **Model:** LightGBM `LGBMRanker` (LambdaRank), `n_estimators=200`, `num_leaves=31`, `learning_rate=0.05`, seed `42`
- **Validation:** leave-one-programme-out CV (45 folds)
- **Features (13):** `cosine_score`, `programme_recall`, `weighted_jaccard`, `overlap_coeff`, `bm25_score`, `n_matched_uris`, `mean_idf_matched`, `mean_idf_job_unmatched`, `count_top_k` (IPF input), `prog_skill_richness`, `job_skill_count`, `prog_implicit_ratio`, `job_implicit_ratio`

## Headline numbers (held-out, 37 / 45 programmes evaluated)

| Metric | Value |
|--------|-------|
| NDCG@10 | 0.759 |
| Precision@5 | 0.324 |
| MRR | 0.750 |
| Coverage@10 | 0.973 |
| Spearman vs hybrid (mean) | −0.332 |

8 programmes had zero consensus positives in their candidate pool and are excluded from the IR metrics (the LOO fold falls back to semantic cosine for those, so the rankings file still has scores for them).

## Feature importance (gain, mean over 37 folds with positive labels)

| # | Feature | Gain |
|---|---------|------|
| 1 | `count_top_k` | 1 456 |
| 2 | `weighted_jaccard` | 729 |
| 3 | `job_skill_count` | 493 |
| 4 | `cosine_score` | 378 |
| 5 | `mean_idf_matched` | 151 |
| 6 | `bm25_score` | 131 |
| 7 | `overlap_coeff` | 128 |
| 8 | `mean_idf_job_unmatched` | 126 |
| 9 | `job_implicit_ratio` | 119 |
| 10 | `programme_recall` | 95 |
| 11 | `prog_skill_richness` | 88 |
| 12 | `prog_implicit_ratio` | 71 |
| 13 | `n_matched_uris` | 38 |

## Interpretation

The strong NDCG@10 on consensus labels obscures a label-design problem.
Three observations together — `count_top_k` dominating importance, negative
Spearman against hybrid, and `job_skill_count` placing third — indicate
that the model is **learning the generalist signal that hybrid's IPF was
designed to suppress**. Cross-strategy consensus is biased toward jobs
that all three strategies happen to surface, and the strategies converge
on the same broad IT job descriptions because their long, generic text
covers everything. The LTR model picks up on `count_top_k` as the most
reliable proxy for "this job appears in many programmes' top-K", which
*is* what consensus says is relevant — but it is the same population that
the hand-tuned formula was working to push *down*.

## Implications

The finding is publishable as-is: it demonstrates that LTR on
unsupervised consensus labels reproduces the very generalist bias the
formula tried to remove, and that consensus-as-relevance has a
construct-validity problem in this corpus. This is a real result, not a
bug.

## Next experiments (not run here)

- **Stricter consensus** — `min_strategies=3` would drop sparse positives
  but should yield more discriminative labels. Worth a side-by-side.
- **Drop `count_top_k`** from the feature set — quantifies how much
  remaining signal the other 12 features carry once the popularity
  shortcut is removed.
- **Apply IPF post-hoc to LTR scores** — keep LTR for retrieval-stage
  ordering, then apply the same two-tier IPF multiplier hybrid uses, so
  the diversity gains from Step 22b carry over.
- **Hand-labelled validation set** — even 10–20 spot-checked
  (programme, job) pairs would give an unbiased read on whether NDCG@10
  against consensus actually tracks human judgement of relevance.

## Decision

Default rankings stay with hybrid (α=0.55, no cross-encoder). LTR is
not adopted. The implementation (`src/alignment/ltr.py`,
`tests/alignment/test_ltr.py`, the regenerable parquet/CSV/JSON
artefacts in this directory, and the `lightgbm` dependency) was
deleted to keep `main` lean. The numbers and feature-importance table
above are the experiment record; if LTR is revisited the module would
need to be reimplemented from this report rather than restored.
