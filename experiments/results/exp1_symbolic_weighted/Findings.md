# IDF + ESCO Reuse-Level Skill Weighting — Findings

## Setup

Replaced uniform skill weights (explicit=1.0, implicit=0.5) in symbolic alignment with a two-factor scheme:

- **Tier weight** based on ESCO `reuseLevel`: transversal=0.3, cross-sector=0.5, sector-specific=0.8, occupation-specific=1.0
- **Corpus IDF** factor: `log(1 + N / df(uri))` where N=345 documents, df=document frequency

Final per-skill weight: `tier_weight(uri) × idf(uri) × (1.0 if explicit, 0.5 if implicit)`

Corpus: 46 programmes × 299 job ads = 13,754 pairs. 365 unique ESCO URIs in the dataset.

## 1. Score distribution changes

| Metric                    | Uniform | Weighted |   Delta |
|---------------------------|---------|----------|---------|
| Jaccard mean (all pairs)  |  0.0621 |   0.0308 | -0.0313 |
| Jaccard median (all)      |  0.0588 |   0.0256 | -0.0332 |
| Jaccard max               |  0.2571 |   0.1731 | -0.0840 |
| Jaccard mean (top-20)     |  0.1297 |   0.0833 | -0.0464 |
| Jaccard median (top-20)   |  0.1316 |   0.0838 | -0.0478 |
| Overlap coeff mean        |  0.3045 |   0.1875 | -0.1170 |
| Overlap coeff median      |  0.2917 |   0.1620 | -0.1297 |
| Skill gap entries (top-20)|  16,359 |   18,177 |  +1,818 |

All scores decreased. The drop is expected: the uniform scheme gave equal credit to transversal skills like "English" and "work in teams" that appear in nearly every document. Removing that shared baseline compresses absolute scores but changes the *relative ordering*.

### Score percentiles

| Percentile | Uniform Jaccard | Weighted Jaccard | Uniform Overlap | Weighted Overlap |
|------------|-----------------|------------------|-----------------|------------------|
| p5         | 0.0089          | 0.0028           | 0.0341          | 0.0115           |
| p25        | 0.0377          | 0.0143           | 0.2000          | 0.0927           |
| p50        | 0.0588          | 0.0256           | 0.2917          | 0.1620           |
| p75        | 0.0833          | 0.0422           | 0.4000          | 0.2480           |
| p95        | 0.1250          | 0.0769           | 0.5833          | 0.4372           |

## 2. Discriminative power

| Strategy          |  CoV (mean ± std) |
|-------------------|-------------------|
| Uniform symbolic  | 0.530 ± 0.085     |
| Weighted symbolic | 0.712 ± 0.118     |
| Semantic          | 0.238 ± 0.025     |
| Hybrid            | 0.082 ± 0.016     |

Coefficient of variation (std/mean per programme) measures how well a strategy separates relevant from irrelevant matches. The weighted scheme has the **highest CoV of all strategies** — a 34% improvement over the uniform scheme — meaning it produces larger relative gaps between top-ranked and bottom-ranked jobs.

## 3. Score spread

| Variant  | Std    | IQR range           |  IQR width |
|----------|--------|---------------------|------------|
| Uniform  | 0.0351 | [0.0377, 0.0833]    |     0.0456 |
| Weighted | 0.0232 | [0.0143, 0.0422]    |     0.0279 |

The absolute spread is narrower, but normalised spread (CoV) is wider — the mean shifted more than the deviation.

## 4. Ranking changes

### Top-1 match agreement

22 of 46 programmes (**47.8%**) retained their top-1 job match. 24 programmes got a different best match under weighting.

Notable top-1 changes:

| Programme | Uniform top-1 | Weighted top-1 | Interpretation |
|-----------|---------------|----------------|----------------|
| Computer games and animation | AI Engineer/Data Scientist (J=0.182) | Game Designer (J=0.137) | Generic AI skills downweighted, domain match promoted |
| Multimedia and Internet Tech | AI Engineer/Data Scientist (J=0.130) | Gameplay Programmer (J=0.116) | Same pattern — domain-specific match rises |
| Digital tech and cyber security | AI Engineer/Data Scientist (J=0.202) | IT syst. and cybersecurity admin (J=0.112) | Security-specific skills now outweigh generic AI overlap |
| Information Systems Cyber Security | Data analyst (J=0.172) | Cybersecurity specialist (J=0.143) | Sector-specific skills correctly promoted |
| Software Systems | Data engineer (J=0.177) | AI Engineer/Data Scientist (J=0.158) | Swap: AI/ML skills valued higher than generic data skills |

### Top-20 set overlap per programme

| Statistic | Value |
|-----------|-------|
| Mean      | 68.3% |
| Min       | 35%   |
| Max       | 90%   |

Programmes with narrower specialisations (e.g. "Information Systems Technology", 90% overlap) were less affected — their skill sets are already specific. Broader programmes ("Information Technologies", 35% overlap) saw the largest reranking because generic skills previously dominated their scores.

### 15 most-changed programmes (lowest top-20 overlap)

| Programme                                  | Uni top-20 J | Wt top-20 J | Top-20 overlap |
|--------------------------------------------|-------------|-------------|----------------|
| Information Technologies                   | 0.1419      | 0.0770      | 35%            |
| Media Technologies                         | 0.1305      | 0.0655      | 40%            |
| Digital Design Technologies                | 0.0959      | 0.0484      | 45%            |
| Software Engineering                       | 0.1360      | 0.0725      | 50%            |
| Marketing Technologies                     | 0.1299      | 0.0724      | 55%            |
| Informatics                                | 0.1310      | 0.0828      | 55%            |
| Informatics Systems                        | 0.1252      | 0.0897      | 55%            |
| Multimedia and Internet Technologies       | 0.0993      | 0.0645      | 55%            |
| Multimedia design                          | 0.0964      | 0.0563      | 55%            |
| Information Technologies                   | 0.0498      | 0.0428      | 60%            |
| Computer games and animation               | 0.1315      | 0.0751      | 60%            |
| Information Systems Engineering            | 0.1440      | 0.0937      | 60%            |
| Programming and Multimedia                 | 0.1023      | 0.0629      | 60%            |
| Information and Communication Technologies | 0.1463      | 0.0925      | 65%            |
| Game Development                           | 0.0842      | 0.0493      | 65%            |

## 5. Cross-strategy correlations

### Spearman rank correlations (mean ± std, 46 programmes)

| Pair             | rho (mean ± std) |
|------------------|------------------|
| uniform↔weighted |  0.908 ± 0.020   |
| uniform���semantic |  0.318 ± 0.078   |
| weighted↔semantic|  0.333 ± 0.085   |
| uniform↔hybrid   |  0.765 ± 0.100   |
| weighted↔hybrid  |  0.675 ± 0.127   |
| semantic↔hybrid  |  0.703 ± 0.089   |

The uniform↔weighted correlation (0.908) confirms that the overall ranking structure is preserved while the top tier is reshuffled. The weighted scheme is slightly more correlated with semantic (0.333 vs 0.318) — a marginal improvement.

### Jaccard@10 — Top-10 job set overlap

| Pair              | J@10 (mean ± std) |
|-------------------|--------------------|
| uniform↔weighted  | 0.438 ± 0.145      |
| uniform↔semantic  | 0.096 ± 0.101      |
| weighted↔semantic | 0.094 ± 0.105      |
| uniform↔hybrid    | 0.222 ± 0.164      |
| weighted↔hybrid   | 0.192 ± 0.183      |
| semantic↔hybrid   | 0.507 ± 0.141      |

The symbolic↔semantic gap (J@10 ~ 0.09) remains large regardless of weighting — the two strategies capture fundamentally different signals.

### Consensus with semantic strategy

| Metric                              | Value   |
|-------------------------------------|---------|
| Uniform top-10 ∩ Semantic top-10    | 1.61 jobs (mean) |
| Weighted top-10 �� Semantic top-10   | 1.57 jobs (mean) |
| Wilcoxon p-value                    | 0.772   |

No significant difference. The weighting scheme does not bring symbolic alignment closer to (or further from) semantic alignment in terms of top-10 set agreement.

## 6. Skill weight examples

### Highest-weighted skills (rare + specific)

| Skill                                | Reuse Level          | Tier | IDF  | Final weight |
|--------------------------------------|----------------------|------|------|--------------|
| application process                  | occupation-specific  | 1.0  | 5.85 | 5.85         |
| manage technical security systems    | occupation-specific  | 1.0  | 4.75 | 4.75         |
| Pascal (computer programming)        | sector-specific      | 0.8  | 5.85 | 4.68         |
| distributed computing                | sector-specific      | 0.8  | 5.85 | 4.68         |
| Maltego                              | sector-specific      | 0.8  | 5.85 | 4.68         |
| develop predictive models            | occupation-specific  | 1.0  | 4.47 | 4.47         |
| network engineering                  | sector-specific      | 0.8  | 5.16 | 4.12         |
| information security strategy        | sector-specific      | 0.8  | 5.16 | 4.12         |
| penetration testing tool             | sector-specific      | 0.8  | 5.16 | 4.12         |
| Swift (computer programming)         | sector-specific      | 0.8  | 4.75 | 3.80         |

### Lowest-weighted skills (common + generic)

| Skill                   | Reuse Level  | Tier | IDF  | Final weight |
|-------------------------|--------------|------|------|--------------|
| English                 | transversal  | 0.3  | 0.99 | 0.30         |
| think creatively        | transversal  | 0.3  | 1.20 | 0.36         |
| think analytically      | transversal  | 0.3  | 1.42 | 0.43         |
| computer technology     | cross-sector | 0.5  | 0.86 | 0.43         |
| manage time             | transversal  | 0.3  | 1.46 | 0.44         |
| Lithuanian              | transversal  | 0.3  | 1.47 | 0.44         |
| plan                    | transversal  | 0.3  | 1.53 | 0.46         |
| assume responsibility   | transversal  | 0.3  | 1.63 | 0.49         |
| lead others             | transversal  | 0.3  | 1.70 | 0.51         |
| communication           | cross-sector | 0.5  | 1.16 | 0.58         |

Weight ratio between the most informative skill ("application process", 5.85) and the least informative ("English", 0.30) is **19.5×**. Under uniform weighting, both contributed 1.0 — identical.

### Reuse level distribution in corpus (365 URIs)

| Level                | Count | Percentage |
|----------------------|-------|------------|
| cross-sector         | 230   | 63.0%      |
| sector-specific      | 81    | 22.2%      |
| transversal          | 43    | 11.8%      |
| occupation-specific  | 10    | 2.7%       |
| unknown              | 1     | 0.3%       |

### IDF distribution (365 URIs, N=345 docs)

| Statistic | Value |
|-----------|-------|
| min       | 0.862 |
| p25       | 2.695 |
| median    | 3.570 |
| p75       | 4.754 |
| max       | 5.846 |
| mean      | 3.713 |
| std       | 1.253 |

## 7. Statistical significance

Wilcoxon signed-rank test on per-programme mean top-20 Jaccard scores:

| Statistic | Value      |
|-----------|------------|
| W         | 0.0        |
| p-value   | 2.84e-14   |

The difference is highly significant — every single programme has lower weighted Jaccard than uniform Jaccard (W=0.0 means all paired differences have the same sign).

## 8. Conclusions

1. **The weighting achieves its primary goal: higher discriminative power.** CoV increased from 0.530 to 0.712 (+34%), the highest of all four strategies. This means the score gap between good and poor matches is relatively wider, making top-ranked results more trustworthy.

2. **Generic skills are correctly downweighted.** "English" (weight 0.30) contributes 19.5× less than "distributed computing" (weight 4.68). Under uniform weighting, sharing "English" and "work in teams" inflated Jaccard for every pair equally — now it does not.

3. **Top-1 matches become more domain-specific.** Programmes like "Computer games and animation" now match to "Game Designer" instead of "AI Engineer" — the weighting correctly identifies that shared gaming-specific skills are more meaningful than shared generic AI skills.

4. **Rankings are moderately reshuffled, not revolutionised.** Spearman rho = 0.908 (overall), top-20 overlap = 68.3% (mean). The broad structure is preserved; the changes are concentrated in the top tier where they matter most.

5. **Cross-strategy agreement is unchanged.** Weighted↔semantic correlation (0.333) is marginally higher than uniform↔semantic (0.318), but the difference is not significant (Wilcoxon p=0.772 on top-10 overlap). The symbolic and semantic strategies capture fundamentally different signals.

6. **Absolute scores are lower**, which means the hybrid formula `α·cosine + (1-α)·jaccard` will need re-tuning. The Jaccard component now operates in a different range (~0.03 mean vs ~0.06), so the optimal alpha will shift toward giving Jaccard more weight. This motivates Step 24 (finer alpha sweep).

7. **The 1,818 additional skill gap entries** (16,359 → 18,177) indicate that the weighting reveals gaps previously hidden by generic skill overlap. When transversal skills no longer mask mismatches, more specific gaps become visible.
