# Extraction Ablation Study — Findings

## Setup

Removed each of the four explicit-extraction modules (S1 NER, S2 PoS, S3 Dictionary, S4 Embedding) one at a time, re-extracted skills, and re-ran symbolic alignment.

Baseline weights: S1=1, S2=1, S3=20, S4=2 (total=24). Threshold=0.35.

## Results

| Config        | Jaccard (all) | Jaccard (top-20) | Overlap (top-20) | Skills/prog | Skills/job | Gap URIs |
|---------------|--------------|-----------------|-----------------|-------------|------------|----------|
| baseline      | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        | 292      |
| no_S1 (NER)   | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        | 292      |
| no_S2 (PoS)   | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        | 292      |
| no_S3 (Dict)  | 0.0634 (-13%)| 0.1199 (-26%)   | 0.3313 (-28%)   | 118.6 (+371%)| 50.5 (+416%)| 1509   |
| no_S4 (Embed) | 0.0725       | 0.1613          | 0.4616          | 25.2        | 9.8        | 292      |

## Analysis

### S3 (Dictionary) is the only effective module

Removing S1, S2, or S4 produces zero change. The ensemble is effectively a dictionary matcher.

**Why:** S3 has weight 20/24. Any S3 hit gets relevance = 20/24 = 0.833, far above threshold 0.35. Candidates that only fire S1+S2+S4 get max relevance = (1+1+2)/24 = 0.167, below threshold. So non-dictionary candidates are always dropped.

### Removing S3 degrades alignment

Without dictionary matching, S4 (embedding similarity) becomes the sole ESCO resolver. This produces ~5x more skills per record (noisy embedding matches) and worse alignment: top-20 Jaccard drops 26%, gap URIs jump from 292 to 1,509.

### Implication

The current weight/threshold configuration makes S1, S2, and S4 redundant. The extraction pipeline is a PhraseMatcher with extra steps.
