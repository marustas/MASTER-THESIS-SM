# Pilot — source-separated propagation + parent-normalisation proxy

Micro-averaged metrics on the 10 pilot documents at each variant, both with raw URIs and with label-cluster normalisation (proxy for ESCO `broaderConcept` rollup):

| Variant | Normalisation | TP | FP | FN | NM | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_canonical_t070` | none | 53 | 468 | 89 | 6 | 0.102 | 0.373 | 0.160 |
| `baseline_canonical_t070` | label_cluster_j0.5 | 57 | 446 | 74 | 0 | 0.113 | 0.435 | 0.180 |
| `same_source_t070` | none | 53 | 468 | 89 | 6 | 0.102 | 0.373 | 0.160 |
| `same_source_t070` | label_cluster_j0.5 | 57 | 446 | 74 | 0 | 0.113 | 0.435 | 0.180 |
| `best_prior_tfidf_s050` | none | 48 | 346 | 94 | 6 | 0.122 | 0.338 | 0.179 |
| `best_prior_tfidf_s050` | label_cluster_j0.5 | 51 | 333 | 80 | 0 | 0.133 | 0.389 | 0.198 |
| `same_source_tfidf_s050` | none | 48 | 347 | 94 | 6 | 0.121 | 0.338 | 0.179 |
| `same_source_tfidf_s050` | label_cluster_j0.5 | 51 | 334 | 80 | 0 | 0.133 | 0.389 | 0.198 |

Variant descriptions:

- `baseline_canonical_t070` — Reference: fixed K=10, τ=0.70, max-cosine scoring (no source mask)
- `same_source_t070` — Source-separated: fixed K=10, τ=0.70, max-cosine scoring
- `best_prior_tfidf_s050` — Reference: adaptive K + TF-IDF score≥0.50 (no source mask)
- `same_source_tfidf_s050` — Source-separated: adaptive K + TF-IDF score≥0.50

Normalisation key:

- `none` — URIs compared as-is (canonical TP/FP/FN definitions)
- `label_cluster_j0.5` — proxy for ESCO parent rollup: greedy single-linkage clustering of all URIs (gold ∪ predicted) by preferred-label token Jaccard ≥ 0.5; canonical per cluster = shortest label. Both sides collapsed before scoring.
