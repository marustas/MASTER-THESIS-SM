# Implicit-extractor variant comparison — pilot results

Micro-averaged metrics on the 10 pilot documents:

| Variant | TP | FP | FN | NM | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_fixedK_t070` | 53 | 468 | 89 | 6 | 0.102 | 0.373 | 0.160 |
| `baseline_fixedK_t050` | 61 | 707 | 81 | 8 | 0.079 | 0.430 | 0.134 |
| `adaptive_k` | 64 | 890 | 78 | 13 | 0.067 | 0.451 | 0.117 |
| `adaptive_k_tfidf_s030` | 53 | 519 | 89 | 9 | 0.093 | 0.373 | 0.148 |
| `adaptive_k_tfidf_s050` | 48 | 346 | 94 | 6 | 0.122 | 0.338 | 0.179 |

Variant descriptions:

- `baseline_fixedK_t070` — Fixed K=10, τ=0.70 (canonical baseline neighbour selection)
- `baseline_fixedK_t050` — Fixed K=10, τ=0.50 (relaxed-threshold baseline from prior sweep)
- `adaptive_k` — Adaptive K: τ_eff = max(0.50, 0.85·cos_max), K_max=25, max-cosine scoring
- `adaptive_k_tfidf_s030` — Adaptive K + TF-IDF score = Σ sim/log(1+|skills|), keep score≥0.30
- `adaptive_k_tfidf_s050` — Same as above but score_threshold=0.50 (more conservative)
