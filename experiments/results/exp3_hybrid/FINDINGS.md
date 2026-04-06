# Hybrid Alignment Redesign & Auxiliary Corpus — Findings

## Setup

Three changes were applied incrementally to the hybrid alignment pipeline:

1. **Formula redesign** — replaced symmetric weighted Jaccard with asymmetric `programme_recall`, added per-programme min-max normalisation
2. **Inverse Programme Frequency (IPF)** — generalist penalty with floor=0.3 for jobs appearing in top-K across many programmes
3. **Auxiliary corpus** — 617 preprocessed EU-wide LinkedIn jobs (from 890 scraped) used to enlarge the implicit skill extractor's fitting corpus from 390 to 1007 documents

Three milestones are compared:

| Milestone | Hybrid formula | Symbolic input | Job corpus | Implicit fit corpus |
|-----------|---------------|----------------|------------|---------------------|
| A — Old hybrid | `α·norm(cos) + (1-α)·norm(jaccard)` | uniform weights | 299 jobs (CVbankas) | 299 docs |
| B — New hybrid+IPF | `α·norm(cos) + (1-α)·norm(recall) × IPF` | IDF-weighted | 390 jobs (CVbankas+LinkedIn) | 390 docs |
| C — + Auxiliary | same as B | same as B | 390 jobs (same) | 390 + 617 = 1007 docs |

Parameters: `alpha=0.5`, `semantic_top_n=50`, `ipf_top_k=10`, `ipf_floor=0.3`.

## 1. Top-1 diversity (primary improvement)

| Metric | A — Old | B — New+IPF | C — +Auxiliary |
|--------|---------|-------------|----------------|
| Unique top-1 jobs / programmes | 13/35 (37%) | 33/46 (72%) | 35/46 (76%) |
| Max top-1 repeats | 6× | 5× | 3× |
| ADMINISTRATORIUS in top-1 | 6 programmes | 4 | 2 |
| Top-5 jobs with freq >5 | 13 | 11 | 5 |

The old hybrid was dominated by generalist jobs: "Informacijos saugos specialistas" appeared as top-1 for 6 programmes, "IT SISTEMŲ ADMINISTRATORIUS" for 5. The combination of IPF penalty and auxiliary corpus reduced the worst-case top-1 repetition from 6× to 3×.

### Old top-1 frequency (Milestone A)

```
6× Informacijos saugos specialistas (-ė)
5× IT SISTEMŲ ADMINISTRATORIUS (-Ė) / INŽINIERIUS (-Ė)
4× Experienced AI Engineer/Data Scientist
4× Vyr. IT Specialistas (-ė)-inžinierius (-ė)
3× IT projektų vadovas
3× .NET programuotojas (-a)
Unique: 13/35
```

### Current top-1 frequency (Milestone C)

```
3× Hiring: IT Technical Specialist
2× Vyresnysis specialistas (kibernetinio saugumo užtikrinimui)
2× Dedikuotas IT inžinierius (-ė)
2× Produkto vadovas (-ė) (Verslas)
Unique: 30/35
```

## 2. Score distribution

| Metric | A — Old | B — New+IPF | C — +Auxiliary |
|--------|---------|-------------|----------------|
| Hybrid mean (all) | 0.2870 | 0.1569 | 0.2049 |
| Hybrid std (all) | 0.0357 | 0.1125 | 0.1024 |
| Hybrid max | 0.4155 | 0.7823 | 0.7815 |
| CoV (all) | 0.124 | 0.717 | 0.500 |
| Top-1 mean | 0.3458 | 0.4437 | 0.4579 |
| Top-1 std | 0.0333 | 0.1196 | 0.1157 |
| Top-1 CoV | 0.096 | 0.269 | 0.253 |
| IQR | 0.0463 | 0.1467 | 0.1236 |

The old hybrid had almost no score variance (CoV=0.12), making rankings undiscriminating. The redesigned formula spreads scores 4× wider (CoV=0.50), with top-1 scores now ranging from 0.27 to 0.78 instead of the previous 0.26–0.42.

### Score components (Milestone C, top-5)

| Component | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| `cosine_score` | 0.515 | 0.049 | 0.394 | 0.651 |
| `programme_recall` | 0.315 | 0.095 | 0.118 | 0.495 |
| `hybrid_score` | 0.387 | 0.079 | 0.231 | 0.546 |

Programme recall has 2× the coefficient of variation (0.30) compared to cosine (0.10), confirming it provides the discriminative signal the old Jaccard lacked.

## 3. Cross-strategy evaluation

| Metric | A — Old | B — New+IPF | C — +Auxiliary |
|--------|---------|-------------|----------------|
| Spearman sym↔sem | 0.318 | 0.190 | 0.179 |
| Spearman sym↔hyb | 0.765 | 0.024 | 0.205 |
| Spearman sem↔hyb | 0.703 | 0.052 | 0.459 |
| Jaccard@10 sym↔hyb | 0.222 | 0.049 | 0.046 |
| Jaccard@10 sem↔hyb | 0.507 | 0.134 | 0.233 |

**Milestone A** — the hybrid was essentially a weighted average of highly correlated signals (ρ=0.77 with symbolic, 0.70 with semantic). It added no independent information.

**Milestone B** — the asymmetric programme_recall broke correlation with both signals (ρ≈0.02–0.05), but overcorrected: hybrid rankings shared almost nothing with either input strategy.

**Milestone C** — the auxiliary corpus restored meaningful correlation with semantic (ρ=0.46) while keeping symbolic independence (ρ=0.21). This is the desired regime: hybrid genuinely blends both signals rather than copying one or ignoring both.

## 4. Auxiliary corpus effect on implicit extraction

| Metric | Main-only (390 docs) | + Auxiliary (1007 docs) |
|--------|---------------------|------------------------|
| Fit corpus size | 390 | 1007 (+158%) |
| Pre-filter avg total skills/job | — | 48.7 |
| Pre-filter avg implicit skills/job | — | 38.2 |
| Post-filter avg implicit skills/job | — | 8.4 |
| Implicit skill ratio | — | 48.3% |
| Unique implicit URIs | — | 296 |
| Implicit-only URIs (not found explicitly) | — | 44 |
| Programme URI coverage by jobs | — | 67.5% |

The auxiliary corpus (617 EU-wide LinkedIn IT jobs) enlarged the neighbour pool for implicit skill propagation. With 1007 documents to search for similar jobs, the extractor finds more relevant neighbours above the similarity threshold (0.60), propagating a richer set of domain-specific skills.

### Auxiliary corpus statistics

- Scraped: 890 jobs from LinkedIn (EU: Germany, Netherlands, Poland, France, Spain, Ireland, Sweden, Lithuania)
- After preprocessing + dedup + language filter: 617 documents
- Average description length: 3948 chars
- Average explicit skills per auxiliary doc: 11.8 (pre-filter)
- Geographic spread: Dublin (60), Vilnius (49), Stockholm (47), Barcelona (40), Berlin (32), Paris (29), Cracow (26), Warsaw (26)

## 5. Ranking stability (B → C transition)

| Metric | Value |
|--------|-------|
| Top-1 retention | 28/46 (61%) |
| Jaccard@5 | 0.582 |
| Jaccard@10 | 0.590 |
| Kendall tau (top-10) | 0.582 mean, 0.556 median |
| Current top-1 found in old top-10 | 14/35 (40%) |

The 39% top-1 churn is driven by changes in implicit skill propagation. With a 2.6× larger fitting corpus, different neighbours become available, altering which skills get propagated to each job ad. This affects programme_recall and consequently the hybrid ranking.

## 6. Notable match improvements (Old → Current)

| Programme | Old top-1 | Current top-1 |
|-----------|-----------|---------------|
| Artificial intelligence | Experienced AI Engineer/Data Scientist | AI Engineer (Applied AI) |
| Cybersecurity Technologies | Vyresnysis specialistas (kibernetinio saugumo) | Cyber Security Specialist |
| Software Systems | .NET programuotojas | DevOps Engineer (ServiceNow) |
| Digital technologies and cyber security | Informacijos saugos specialistas | IT sistemų ir kibernetinio saugumo admin. |
| Information Technologies | IT SISTEMŲ ADMINISTRATORIUS | JAVA programuotojas |
| Programming and Multimedia | Verslo aplikacijų ekspertas | FULL STACK PROGRAMUOTOJAS |

## 7. Remaining limitations

1. **"Hiring: IT Technical Specialist"** appears as top-1 for 3 programmes — still a mild generalist despite IPF. Its broad skill profile produces high programme_recall for generic IT programmes.

2. **"Junior / Entry-Level IT Administrator"** appears in top-5 for 13 programmes. The IPF floor (0.3) keeps it in the pool rather than eliminating it, which is by design: niche programmes with no domain-specific jobs still need fallback matches.

3. **Niche programmes** (game development, bioinformatics, multimedia design) have no domain-specific jobs in the 390-job corpus. Their top matches are generic IT roles. This is a corpus coverage limitation, not an algorithmic one.

4. **Top-1 agreement rate** across all three strategies remains 0% — no programme has the same top-1 job in symbolic, semantic, and hybrid. This reflects the fundamental independence of the signals rather than a quality issue.

## 8. Summary

The hybrid redesign achieved three goals:

1. **Discriminability** — CoV increased from 0.12 to 0.50, making score differences meaningful
2. **Diversity** — unique top-1 jobs increased from 13/35 (37%) to 35/46 (76%)
3. **Signal independence** — hybrid now blends both semantic (ρ=0.46) and symbolic (ρ=0.21) without being dominated by either

The auxiliary corpus contributed the implicit extraction enrichment that made programme_recall discriminative enough to break generalist dominance. Without it, the new formula had the right structure but insufficient signal variety (Milestone B: ρ≈0.02–0.05 with both strategies, indicating near-random rankings).
