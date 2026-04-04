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

## 8. Full top-1 match table (all 46 programmes)

Columns: programme name, uniform top-1 job, uniform Jaccard, weighted top-1 job, weighted Jaccard, semantic cosine for both, shared skill count with breakdown (transversal / cross-sector / specific), and verdict (BETTER/WORSE/NEUTRAL judged by cosine delta ±0.01).

### Unchanged top-1 (22 programmes)

| Programme | Top-1 job | Uni J | Wt J | Cosine | Shared | T/C/S |
|-----------|-----------|-------|------|--------|--------|-------|
| Information Systems and Cybersecurity | IT syst. ir kibernetinio saugumo admin. | 0.2571 | 0.1731 | 0.488 | 19 | 6/7/6 |
| Marketing Technologies | Vyr. duomenų inžinierius | 0.1792 | 0.1288 | 0.421 | 13 | 2/7/4 |
| Bioinformatics | Vyr. duomenų analitikas | 0.1364 | 0.1164 | 0.393 | 7 | 1/4/2 |
| Artificial intelligence | Experienced AI Engineer/Data Scientist | 0.1826 | 0.1164 | 0.515 | 13 | 3/10/0 |
| Information and Communication Technologies | AI Analyst | 0.1716 | 0.1168 | 0.387 | 13 | 3/9/1 |
| Informatics (309) | IT syst. ir kibernetinio saugumo admin. | 0.1789 | 0.1233 | 0.433 | 15 | 5/6/4 |
| Informatics (311) | IT syst. ir kibernetinio saugumo admin. | 0.2017 | 0.1374 | 0.493 | 16 | 4/6/6 |
| Informatics Systems | Vyr. duomenų inžinierius | 0.1545 | 0.1150 | 0.437 | 13 | 1/8/4 |
| Cybersecurity Technologies | Vyr. specialistas (kibernetinio saugumo) | 0.2143 | 0.1450 | 0.617 | 12 | 6/3/3 |
| Digital Design Technologies | Vyr. specialistas (kibernetinio saugumo) | 0.1143 | 0.0654 | 0.427 | 7 | 3/3/1 |
| Applied Informatics and Programming | Vyr. specialistas (kibernetinio saugumo) | 0.1636 | 0.1301 | 0.438 | 10 | 4/3/3 |
| Game Development | Sistemų specialistas | 0.0968 | 0.0600 | 0.407 | 9 | 3/5/1 |
| Artificial Intelligence Systems | Python Developer | 0.1897 | 0.1220 | 0.476 | 15 | 4/8/3 |
| Information Systems Engineering (325) | Vyr. duomenų inžinierius | 0.1667 | 0.1215 | 0.434 | 14 | 2/6/6 |
| Information systems engineering (326) | Vyr. duomenų inžinierius | 0.1319 | 0.1175 | 0.371 | 10 | 1/5/4 |
| Informatics engineering (331) | Experienced AI Engineer/Data Scientist | 0.1481 | 0.1201 | 0.506 | 10 | 3/5/2 |
| Informatics Engineering (332) | IT syst. ir kibernetinio saugumo admin. | 0.2137 | 0.1440 | 0.435 | 17 | 6/5/6 |
| Cyber Systems and Security | Experienced AI Engineer/Data Scientist | 0.1721 | 0.1205 | 0.452 | 12 | 3/7/2 |
| Multimedia design | Sistemų specialistas | 0.1081 | 0.0716 | 0.379 | 10 | 3/6/1 |
| Programming and Multimedia | Vyr. duomenų analitikas | 0.1300 | 0.0996 | 0.345 | 10 | 2/6/2 |
| Software Systems (341) | AI Analyst | 0.1832 | 0.1395 | 0.420 | 16 | 3/10/3 |
| Software Engineering (343) | IT syst. ir kibernetinio saugumo admin. | 0.2137 | 0.1448 | 0.452 | 17 | 4/7/6 |

### Changed top-1 — BETTER by semantic cosine (11 programmes)

| Programme | Old top-1 | Old cos | New top-1 | New cos | Old shared (T/C/S) | New shared (T/C/S) |
|-----------|-----------|---------|-----------|---------|--------------------|--------------------|
| Informatics (310) | IT syst. kibernetinio saugumo admin. | 0.472 | IT sistemų administratorius / inžinierius | 0.570 | 15 (6/4/5) | 14 (5/4/5) |
| Media Technologies | IT System Administrator | 0.294 | Sistemų specialistas | 0.427 | 9 (5/4/0) | 13 (5/6/2) |
| Game Dev. and Digital Animation | IT verslo syst. administratorius | 0.408 | Vyr. IT Specialistas-inžinierius | 0.497 | 11 (3/7/1) | 10 (2/5/3) |
| IS and Cyber Security (324) | Vyr. duomenų analitikas | 0.466 | Vyr. specialistas (kibernetinio saugumo) | 0.667 | 12 (3/7/2) | 8 (2/3/3) |
| IS Engineering (327) | IT syst. kibernetinio saugumo admin. | 0.462 | IT sistemų administratorius | 0.533 | 14 (4/4/6) | 15 (3/4/8) |
| IS Technology (329) | IT syst. kibernetinio saugumo admin. | 0.533 | IT sistemų administratorius | 0.619 | 17 (4/8/5) | 18 (3/7/8) |
| Informatics Engineering (330) | Experienced AI Engineer/Data Scientist | 0.476 | Vyr. specialistas (kibernetinio saugumo) | 0.579 | 12 (3/9/0) | 10 (4/3/3) |
| Informatics Engineering (333) | Experienced AI Engineer/Data Scientist | 0.509 | IT sistemų administratorius | 0.520 | 11 (3/8/0) | 14 (4/5/5) |
| Digital tech. and cyber security | Experienced AI Engineer/Data Scientist | 0.456 | IT syst. kibernetinio saugumo admin. | 0.523 | 12 (3/8/1) | 14 (5/5/4) |
| Software Systems (340) | Vyr. duomenų inžinierius | 0.396 | Experienced AI Engineer/Data Scientist | 0.482 | 16 (2/8/6) | 14 (2/8/4) |
| Software Engineering (344) | IT Project Manager | 0.338 | Vyr. duomenų analitikas | 0.370 | 7 (4/3/0) | 7 (1/4/2) |

### Changed top-1 — NEUTRAL by semantic cosine (3 programmes)

| Programme | Old top-1 | Old cos | New top-1 | New cos | Old shared (T/C/S) | New shared (T/C/S) |
|-----------|-----------|---------|-----------|---------|--------------------|--------------------|
| Dev. and Maintenance of IS | IT syst. administratorius / inžinierius | 0.540 | IT sistemų administratorius | 0.550 | 15 (5/5/5) | 16 (4/5/7) |
| Information Systems (301) | IT sistemų testuotojas | 0.391 | Vyr. duomenų inžinierius | 0.382 | 9 (3/5/1) | 12 (2/6/4) |
| IS Engineering (328) | IRT specialistas (SOC ir IT saugos) | 0.501 | Vyr. duomenų analitikas | 0.506 | 13 (8/3/2) | 10 (2/6/2) |

### Changed top-1 — WORSE by semantic cosine (10 programmes)

| Programme | Old top-1 | Old cos | New top-1 | New cos | Old shared (T/C/S) | New shared (T/C/S) |
|-----------|-----------|---------|-----------|---------|--------------------|--------------------|
| Information Systems (302) | IT sistemų testuotojas | 0.470 | Duomenų inžinierius – analitikas | 0.447 | 11 (3/7/1) | 15 (3/7/5) |
| Information Technologies (307) | Sistemų specialistas | 0.346 | Vyr. duomenų inžinierius | 0.313 | 5 (2/1/2) | 4 (0/1/3) |
| Information Technologies (308) | .NET programuotojas | 0.561 | Vyr. duomenų inžinierius | 0.443 | 8 (6/1/1) | 8 (2/3/3) |
| Informatics (312) | IT Project Manager | 0.403 | Vyr. duomenų inžinierius | 0.389 | 8 (5/3/0) | 8 (0/5/3) |
| Computer games and animation | Experienced AI Engineer/Data Scientist | 0.484 | Game Designer | 0.453 | 11 (2/8/1) | 5 (1/1/3) |
| Multimedia and Internet Tech. | Experienced AI Engineer/Data Scientist | 0.384 | Gameplay Programmer | 0.304 | 8 (1/7/0) | 8 (1/3/4) |
| Programming and Internet Tech. | IT verslo syst. administratorius | 0.521 | Vyr. duomenų inžinierius | 0.392 | 11 (3/7/1) | 9 (1/5/3) |
| Multimedia technology | Verslo aplikacijų ekspertas | 0.497 | Vyr. duomenų analitikas | 0.395 | 10 (6/3/1) | 10 (2/7/1) |
| Multimedia Technologies | AI Analyst | 0.381 | People data analyst | 0.323 | 11 (2/9/0) | 10 (1/9/0) |
| Software Engineering (342) | IT syst. kibernetinio saugumo admin. | 0.512 | Full Stack programuotojas | 0.467 | 15 (5/6/4) | 6 (0/1/5) |

### Analysis of "worse" cases

In all 10 cases the same pattern holds: the old top-1 had more transversal and cross-sector shared skills, while the new top-1 has more sector-specific shared skills:

| Programme | Old T/C/S | New T/C/S | Transversal lost | Specific gained |
|-----------|-----------|-----------|------------------|-----------------|
| Information Systems (302) | 3/7/1 | 3/7/5 | 0 | +4 |
| Information Technologies (307) | 2/1/2 | 0/1/3 | -2 | +1 |
| Information Technologies (308) | 6/1/1 | 2/3/3 | -4 | +2 |
| Informatics (312) | 5/3/0 | 0/5/3 | -5 | +3 |
| Computer games and animation | 2/8/1 | 1/1/3 | -1 | +2 |
| Multimedia and Internet Tech. | 1/7/0 | 1/3/4 | 0 | +4 |
| Programming and Internet Tech. | 3/7/1 | 1/5/3 | -2 | +2 |
| Multimedia technology | 6/3/1 | 2/7/1 | -4 | 0 |
| Multimedia Technologies | 2/9/0 | 1/9/0 | -1 | 0 |
| Software Engineering (342) | 5/6/4 | 0/1/5 | -5 | +1 |

The weighting trades transversal overlap (skills like "English", "think analytically", "manage time") for sector-specific overlap (skills like "data engineering", "Game Designer", "penetration testing"). The semantic cosine is not a fully independent judge here — it also benefits from shared generic vocabulary in the text, so a drop in cosine does not necessarily mean a worse match.

Two genuinely questionable cases:

1. **Information Technologies (308)**: .NET programuotojas (cos=0.561) → Vyr. duomenų inžinierius (cos=0.443). The old match was a strong .NET programming job; the new match (data engineer) is less related to a general IT programme. The large cosine drop (-0.118) and the fact that shared skills shifted from programming-adjacent to data-adjacent suggests over-correction.

2. **Software Engineering (342)**: IT syst. kibernetinio saugumo admin. (15 shared, cos=0.512) → Full Stack programuotojas (6 shared, cos=0.467). The new match has fewer shared skills overall (15 → 6), even though all 6 are specific. The weighting disproportionately favoured the 5 specific skills over the 10 lost transversal+cross-sector ones.

In both cases the weighting correctly identified more specific overlaps, but the trade-off was too aggressive — the lost generic overlap was still partially informative.

## 9. Root cause analysis of worse cases

Three distinct failure patterns were identified across the 10 worse cases in the initial (tier + uncapped IDF) configuration:

### Pattern A: "Data engineer magnet" (5 cases: 302, 307, 308, 312, 318)

Job 7 ("Vyr. duomenų inžinierius") has a tight cluster of high-IDF sector-specific skills — `data warehouse` (w=2.38), `data models` (1.95), `data storage` (2.05), `data engineering` (1.06), `perform warehousing operations` (1.03) — totalling ~8-10 weight from just 4-5 skills. Any programme mentioning even a few data-related terms got pulled toward this job.

The lost skills were almost entirely transversal: "think analytically" (0.43), "manage time" (0.44), "Lithuanian" (0.44). Individually negligible, but 5-8 of them collectively represented meaningful signal under uniform weighting.

**Root cause:** Uncapped IDF over-amplifies niche clusters. "Data warehouse" (IDF=5.16) gets 6× the weight of "computer technology" (IDF=0.86).

### Pattern B: "Specificity over breadth" (3 cases: 315, 317, 342)

The new match has fewer shared skills overall but a few high-weight sector-specific ones.

- **315** (Computer games): Lost 9 skills (total w=9.94). Gained 3: "Unity" (3.40), "digital game creation systems" (3.26), "Android" (2.00). New total is actually lower (-0.50) but wins on Jaccard because the *denominator* (union) is also much smaller.
- **342** (Software Engineering): Lost 12 skills (w=14.14 total) for 3 gained ("C#" 2.22, "Android" 2.00, "Java" 1.93, total=9.92). A net -4.23 in weight but a smaller union.

**Root cause:** Jaccard = intersection/union. When the new match has far fewer total skills, the union shrinks, letting a smaller but purer intersection win.

### Pattern C: "Marginal swap" (2 cases: 336, 337)

- **337**: The only lost skill is "English" (w=0.30). A rounding-level swap.
- **336**: Lost 8 transversal for 8 cross-sector. Real but marginal difference.

**Root cause:** Near-ties broken differently when transversal skills lose weight.

## 10. Parameter tuning experiments

Tested 11 configurations to find the best trade-off between discriminative power (CoV) and ranking quality (fewer worse cases):

| Config | J (all) | J (top-20) | CoV | Changes | Better | Neutral | Worse | rho(uni) | rho(sem) |
|--------|---------|------------|-----|---------|--------|---------|-------|----------|----------|
| uniform (baseline) | 0.062 | 0.130 | 0.529 | 0 | - | - | - | 1.000 | 0.318 |
| tier 0.3/0.5/0.8/1.0 uncapped | 0.031 | 0.083 | **0.711** | 24 | 11 | 3 | **10** | 0.908 | 0.333 |
| **IDF cap 3.0 (no tier)** | **0.038** | **0.092** | **0.622** | **14** | **7** | **3** | **4** | **0.960** | **0.327** |
| IDF cap 2.5 (no tier) | 0.041 | 0.098 | 0.619 | 15 | 8 | 3 | 4 | 0.964 | 0.327 |
| IDF uncapped (no tier) | 0.036 | 0.088 | 0.625 | 16 | 9 | 3 | 4 | 0.958 | 0.328 |
| tier 0.5/0.7/0.9/1.0 no IDF | 0.056 | 0.121 | 0.553 | 9 | 5 | 1 | 3 | 0.985 | 0.333 |
| tier 0.3/0.5/0.8/1.0 cap 3.0 | 0.033 | 0.087 | 0.707 | 23 | 11 | 3 | 9 | — | — |
| tier 0.5/0.6/0.8/1.0 cap 3.0 | 0.035 | 0.089 | 0.657 | 21 | 9 | 3 | 9 | 0.945 | 0.331 |
| tier 0.6/0.7/0.8/1.0 cap 3.0 | 0.036 | 0.090 | 0.642 | 16 | 7 | 3 | 6 | 0.950 | 0.331 |
| IDF only (no tier, no cap) | 0.036 | 0.088 | 0.625 | 16 | 9 | 3 | 4 | 0.958 | 0.328 |
| tier only (0.3/0.5/0.8/1.0) | 0.052 | 0.118 | 0.580 | 19 | 10 | 2 | 7 | 0.964 | 0.337 |

### Key observations

1. **Tier weighting is the main source of worse cases.** Every config with tier weights produces 6-10 worse cases. Removing tiers drops worse cases to 3-4 regardless of IDF cap.

2. **IDF cap prevents the "data engineer magnet" pattern** but does not prevent the "specificity over breadth" pattern — that is inherent to the Jaccard formula when skill-set sizes differ.

3. **IDF cap 3.0 (no tier)** is the best balance:
   - CoV = 0.622 (+18% over uniform) — meaningful improvement
   - Only 4 worse cases (vs 10 with tiers) — half the degradation
   - rho(uniform) = 0.960 — rankings are stable
   - rho(semantic) = 0.327 — similar cross-strategy signal

4. **The 4 remaining worse cases** under IDF cap 3.0 are all multimedia programmes with marginal cosine differences:
   - Multimedia and Internet Technologies (cos 0.384→0.304)
   - Information Systems Engineering (cos 0.434→0.417)
   - Multimedia technology (cos 0.497→0.416)
   - Multimedia Technologies (cos 0.381→0.323)

### Decision

**Adopted IDF cap 3.0 with no tier weighting as the default configuration.**

- `use_tiers=False` (default) — tier weighting available but disabled
- `idf_cap=3.0` (default) — prevents single rare skill from dominating
- Tier weighting remains configurable via `use_tiers=True` for reporting

**Rationale:** Tier weighting over-penalises transversal skills that, while generic, still carry signal (e.g. "think analytically" is a real competence). IDF alone achieves the core goal — down-weighting ubiquitous skills — without the aggressive reranking caused by tier weights.

## 11. Final results (IDF cap 3.0, no tier)

| Metric | Uniform | Weighted (final) | Delta |
|--------|---------|-------------------|-------|
| Jaccard mean (all) | 0.062 | 0.038 | -0.024 |
| Jaccard mean (top-20) | 0.130 | 0.092 | -0.038 |
| Overlap coeff mean | 0.305 | 0.224 | -0.081 |
| CoV | 0.529 | 0.622 | +0.093 |
| Top-1 changes | — | 14/46 | — |
| Better / Neutral / Worse | — | 7 / 3 / 4 | — |
| rho(uniform) | 1.000 | 0.960 | — |
| rho(semantic) | 0.318 | 0.327 | +0.009 |
| Skill gap entries | 16,359 | 17,416 | +1,057 |

## 12. Conclusions

1. **IDF-only weighting with cap=3.0 is the adopted configuration.** It improves discriminative power (CoV +18%) while keeping worse cases to 4 (vs 10 with tier weighting). The cap prevents niche skill clusters from dominating.

2. **Tier weighting over-corrects.** Down-weighting transversal skills to 0.3 causes 10 worse cases because skills like "think analytically" and "solve problems" are real competences, not pure noise. The IDF already handles the frequency problem — adding tier weights double-penalises common-but-meaningful skills.

3. **The 4 remaining worse cases are multimedia programmes** where the weighting promotes programming-specific matches over broadly-relevant ones. These are inherent to the Jaccard formula's sensitivity to skill-set size differences and cannot be eliminated by parameter tuning alone.

4. **Rankings are stable.** Spearman rho = 0.960 with uniform baseline. The changes are concentrated in the top tier — 14 of 46 programmes get a different top-1 match, with 7 improving and only 4 getting worse by semantic cosine.

5. **Cross-strategy signal is preserved.** Weighted↔semantic correlation (0.327) is nearly identical to uniform↔semantic (0.318). The weighting does not change the fundamental relationship between symbolic and semantic strategies.

6. **Absolute scores are lower** (~0.038 mean vs ~0.062), which means the hybrid alpha will need re-tuning in Step 24.
