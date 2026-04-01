# Extraction Ablation Study — Findings

## Setup

Removed each of the four explicit-extraction modules (S1 NER, S2 PoS, S3 Dictionary, S4 Embedding) one at a time, re-extracted skills, and re-ran symbolic alignment.

Baseline weights: S1=1, S2=1, S3=20, S4=2 (total=24). Threshold=0.35.

## 1. Standard ablation (paper weights)

| Config        | Jaccard (all) | Jaccard (top-20) | Overlap (top-20) | Skills/prog | Skills/job | Gap URIs |
|---------------|--------------|-----------------|-----------------|-------------|------------|----------|
| baseline      | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        | 292      |
| no_S1 (NER)   | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        | 292      |
| no_S2 (PoS)   | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        | 292      |
| no_S3 (Dict)  | 0.0634 (-13%)| 0.1199 (-26%)   | 0.3313 (-28%)   | 118.6 (+371%)| 50.5 (+416%)| 1509   |
| no_S4 (Embed) | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        | 292      |
| s3_only       | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        | 292      |

S3-only produces identical results to the full 4-module baseline, confirming S1/S2/S4 are redundant.

## 2. Why S1, S2, S4 contribute nothing

Per-document analysis of 10 sample records shows:

- S1 (NER) produces 0–25 unique candidates per document (not found by S3)
- S2 (PoS) produces 9–60 unique candidates per document
- Their max relevance scores are 0.107–0.118 — far below the 0.35 threshold

The mathematical ceiling for a non-S3 candidate is:
- S1+S4 only: `(1*1.0 + 2*1.0) / 24 = 0.125` (even with perfect S4 similarity)
- S2+S4 only: `(1*1.0 + 2*1.0) / 24 = 0.125`
- S1+S2+S4: `(1*1.0 + 1*1.0 + 2*1.0) / 24 = 0.167`

All below 0.35. No non-dictionary candidate can ever pass the threshold with these weights.

## 3. Rebalanced weight exploration

Tested alternative weight schemes to see if S1/S2/S4 can be made useful:

| Config              | Weights (S1,S2,S3,S4) | Jaccard (all) | Jaccard (top-20) | Overlap (top-20) | Skills/prog | Skills/job |
|---------------------|----------------------|--------------|-----------------|-----------------|-------------|------------|
| paper_default       | 1, 1, 20, 2         | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        |
| equal_weights       | 1, 1, 1, 1          | 0.0623       | 0.1187          | 0.3154          | 116.8       | 47.9       |
| rebalanced          | 2, 2, 5, 3          | 0.0679       | 0.1329          | 0.3622          | 61.3        | 23.3       |
| s3_only             | 0, 0, 1, 0          | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        |
| no_s3_rebalanced    | 3, 2, 0, 5          | 0.0642       | 0.1200          | 0.3313          | 120.7       | 51.3       |

Every configuration that reduces S3 dominance or removes S3 produces worse alignment. Adding S1/S2/S4 candidates introduces noise — more skills per record, lower Jaccard, more gap URIs.

## 4. Conclusions

1. **S3 (ESCO Dictionary) is the sole effective extraction module.** The paper's weight scheme (S3=20) ensures only dictionary-matched terms pass the relevance threshold. This is not a bug — it produces the best alignment quality.

2. **S1 and S2 serve an architectural role only.** They generate candidate phrases that *could* be useful if they also happen to be S3 hits (boosting their S3 score). But since S3 already fires independently via PhraseMatcher, the S1/S2 contribution is redundant.

3. **S4 (Embedding) acts as a URI resolver, not a skill discoverer.** When S3 is present, S4's role is to assign ESCO URIs to candidates. When S3 is absent, S4 over-matches, producing 5x more noisy skills.

4. **The paper's weight design is intentional.** Gugnani & Misra (2020) used ONet=20 + Hope=10 + Wikipedia=20 for dictionaries (total=50) vs NER=1, PoS=1, W2V=2. Their design also heavily favoured dictionary matching. Our ESCO adaptation (S3=20) mirrors this intent.

5. **For this dataset, the extraction pipeline could be simplified to S3 alone** without any loss in alignment quality. The other modules add computational cost (~4x slower) with zero benefit.
