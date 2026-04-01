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

## 4. Extraction improvement experiments

Two improvements were implemented and tested:

**S4 soft matching** — accept non-S3 candidates with high S4 embedding similarity as a second discovery pathway. Bypasses the relevance formula when S4 cosine sim exceeds a threshold.

**S4 context filter** — reject S3 dictionary hits with low S4 contextual similarity (catches spurious matches like "Java" in a travel context).

### S4 soft matching results

| Config         | Jaccard (all) | Jaccard (top-20) | Overlap (top-20) | Skills/prog | Skills/job |
|----------------|--------------|-----------------|-----------------|-------------|------------|
| baseline       | 0.0725       | 0.16132         | 0.46158         | 25.2        | 9.8        |
| soft_0.85      | 0.0689       | 0.14911         | 0.42916         | 31.3        | 11.4       |
| soft_0.80      | 0.0687       | 0.14470         | 0.39958         | 39.6        | 14.6       |
| soft_0.75      | 0.0713       | 0.14186         | 0.38813         | 55.2        | 19.8       |
| soft_0.70      | 0.0710       | 0.13904         | 0.37637         | 70.9        | 25.7       |

Every soft-match threshold degrades alignment. Even at 0.85 (very strict), top-20 Jaccard drops 7.6%.

**Root cause — noisy soft matches:** Inspection of added skills shows the embedding model maps short text fragments to semantically distant ESCO labels: "varnish" → "transfer varnish", "tools" → "machine tools", "scripts" → "read scripts", "sports" → "exercise sports", "office" → "office administration". The MiniLM-L6 model lacks the precision to distinguish ICT contexts from generic word similarity at the single-term level.

### S4 context filter results

| Config    | Jaccard (all) | Jaccard (top-20) | Skills/prog |
|-----------|--------------|-----------------|-------------|
| baseline  | 0.0725       | 0.16132         | 25.2        |
| ctx_0.25  | 0.0725       | 0.16132         | 25.2        |
| ctx_0.30  | 0.0725       | 0.16132         | 25.2        |
| ctx_0.35  | 0.0726       | 0.16132         | 25.2        |
| ctx_0.40  | 0.0550       | 0.14079         | 24.8        |

No effect below 0.40 — virtually all S3 dictionary hits have S4 similarity above 0.35, meaning the PhraseMatcher rarely produces spurious matches. At 0.40 it starts removing valid hits and hurts alignment.

### Combined (soft + context)

No additive benefit. The context filter removes too few S3 hits to offset the noise from soft matching.

### Overlap analysis (10 programmes × 10 jobs)

| Config   | Prog skills | Job skills | Overlap |
|----------|-------------|------------|---------|
| baseline | 101         | 74         | 25      |
| soft_0.80| 151         | 111        | 46      |

Soft matching increases skill-set overlap (+84%), but the added noise (+50 programme, +37 job skills) dilutes Jaccard more than the overlap helps.

## 5. Conclusions

1. **S3 (ESCO Dictionary) is the sole effective extraction module.** The paper's weight scheme (S3=20) ensures only dictionary-matched terms pass the relevance threshold. This is not a bug — it produces the best alignment quality.

2. **S1 and S2 serve an architectural role only.** They generate candidate phrases that *could* be useful if they also happen to be S3 hits (boosting their S3 score). But since S3 already fires independently via PhraseMatcher, the S1/S2 contribution is redundant.

3. **S4 (Embedding) acts as a URI resolver, not a skill discoverer.** When S3 is present, S4's role is to assign ESCO URIs to candidates. When S3 is absent, S4 over-matches, producing 5x more noisy skills.

4. **The paper's weight design is intentional.** Gugnani & Misra (2020) used ONet=20 + Hope=10 + Wikipedia=20 for dictionaries (total=50) vs NER=1, PoS=1, W2V=2. Their design also heavily favoured dictionary matching. Our ESCO adaptation (S3=20) mirrors this intent.

5. **For this dataset, the extraction pipeline could be simplified to S3 alone** without any loss in alignment quality. The other modules add computational cost (~4x slower) with zero benefit.

6. **Extraction improvements do not help with the current embedding model.** S4 soft matching adds noise because MiniLM-L6 cannot reliably distinguish ICT skill terms from generic words at the single-token level. S4 context filtering has no effect because S3 (PhraseMatcher) rarely produces spurious matches — exact string matching is already precise.

7. **The real bottleneck is not extraction but alignment scoring.** The skill sets are clean (S3-only is precise); the low Jaccard (0.072) reflects genuine vocabulary mismatch between programme descriptions and job ads. Step 23 (IDF + reuse-level weighting) addresses this more directly by weighting skills by informativeness.
