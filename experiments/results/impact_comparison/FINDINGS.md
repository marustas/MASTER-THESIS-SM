# Impact Comparison — Steps 27 & 31

Evaluated the impact of two features from the last session on hybrid alignment quality, tested individually and combined against a baseline (no skill embeddings, no programme IDF).

## Configurations tested

| Config | Description |
|--------|-------------|
| A — Baseline | No skill embeddings, no programme IDF (previous pipeline default) |
| B — +DescEmb | ESCO description embeddings for coherence boost (Step 27, delta=0.2) |
| C — +ProgIDF | Programme-level IDF in symbolic refinement (Step 31) |
| D — +Both | Description embeddings + programme IDF combined |

## Summary metrics

| Metric | Baseline | +DescEmb | +ProgIDF | +Both |
|--------|----------|----------|----------|-------|
| Top-1 unique jobs | 35 | 35 | **37** | **38** |
| Top-1 diversity | 0.778 | 0.778 | **0.822** | **0.844** |
| Top-1 score mean | 0.286 | 0.284 | **0.304** | **0.303** |
| Top-1 score CoV | 0.412 | 0.408 | **0.383** | **0.384** |
| Top-5 Jaccard vs baseline | — | 0.941 | 0.814 | 0.821 |
| Kendall tau (shared top-5) | — | 0.914 | 0.679 | 0.662 |

## Score direction (top-1 per programme)

| | Improved (>+0.001) | Degraded (<-0.001) | Stable | Mean delta |
|---|---|---|---|---|
| **+DescEmb** | 8 (18%) | **24 (53%)** | 13 (29%) | **-0.002** |
| **+ProgIDF** | **27 (60%)** | 10 (22%) | 8 (18%) | **+0.018** |
| **+Both** | **28 (62%)** | 9 (20%) | 8 (18%) | **+0.017** |

## Key finding: description embeddings removed

ESCO description embeddings (Step 27) were intended to improve the coherence boost — rewarding matches where the overlapping skills form a semantically coherent cluster. The hypothesis was that 1-3 sentence ESCO descriptions would produce more meaningful pairwise cosine similarity than 2-3 word skill labels.

**Result:** The feature actively degraded results. 53% of programmes saw lower top-1 scores, while only 18% improved. The coherence boost (delta=0.2 x mean pairwise cosine) introduced noise rather than signal — the pairwise similarity between ESCO descriptions was not discriminative enough to distinguish coherent from incoherent skill clusters at this scale.

**Decision:** Removed the coherence boost entirely from `compute_match_quality()` and deleted `build_skill_description_embeddings()`, `save_skill_embeddings()`, and `_load_skill_embeddings()`. The match quality multiplier now uses only specificity ratio and generic penalty.

## Key finding: programme IDF enabled

Programme-level IDF (Step 31) weights each programme's skills by distinctiveness relative to other programmes. A skill unique to one programme carries more weight than a skill shared across all programmes. Job skills still use corpus-wide IDF.

**Result:** 60% of programmes improved, with a +0.018 mean score lift. The changes are qualitatively sensible — generic roles replaced by domain-specific ones:

- "Information Systems & Cyber Security": vuln testing specialist -> SOC analyst for threat hunting (+0.044)
- "Development & Maintenance of IS": cyber security manager -> IT department administrator (+0.045)
- "Information Systems": IT administrator -> business systems & process analyst (+0.030)
- "Digital tech & cyber security": cyber security manager -> AI engineer (+0.028)

One minor regression: "Cyber Systems & Security" swapped IT Compliance Manager for Cyber Security Manager (-0.010), which is arguably a lateral move.

**Decision:** Enabled `use_programme_idf=True` as the default in both `align_hybrid()` and the pipeline.

## Changes applied

1. Removed `skill_embeddings`, `delta`, `min_coherence_skills` parameters from `align_hybrid()` and `run_hybrid_alignment()`
2. Removed coherence boost from `compute_match_quality()` — now returns only `specificity_ratio`, `generic_penalty`, `quality_multiplier`
3. Removed `build_skill_description_embeddings()` and `save_skill_embeddings()` from `skill_weights.py`
4. Removed `_load_skill_embeddings()` and `SKILL_EMBEDDINGS_PATH` from `hybrid.py`
5. Set `use_programme_idf=True` as default in `align_hybrid()` and pipeline step 10
6. Removed 11 coherence/embedding-related tests (478 tests remaining, all passing)
