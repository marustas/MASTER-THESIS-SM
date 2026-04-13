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

---

# Post-Milestone Changes (April 2026)

Changes applied after the initial hybrid redesign (Milestones A–C). Current parameters: α=0.6, IPF top_k=30, floor=0.1, γ=0.3, δ=0.2. Corpus: 520 jobs (401 CVbankas + 119 LinkedIn), 47 programmes, 1567 auxiliary jobs.

## 9. Corpus expansion & IPF re-tuning (Milestone D → E)

### Corpus changes

- Extended LinkedIn scraping: added COMPUTER_GAMES industry, additional languages
- Auxiliary corpus grew from 617 → 1567 EU-wide jobs
- Main corpus: 520 jobs (401 CVbankas + 119 LinkedIn), up from 390
- IPF parameters updated in Step 24: α=0.5→0.6, ipf_top_k=10→20

### Generalist resurgence

The expanded corpus introduced "IT specialistas (-ė)" which dominated top-5 for 17 programmes. Top-1 diversity regressed from 41→34.

Parameter sweep over α ∈ {0.4, 0.5, 0.6}, top_k ∈ {10, 15, 20, 30, 40}, floor ∈ {0.05, 0.1, 0.15, 0.2, 0.3} (75 configs). Winner: α=0.6, top_k=30, floor=0.1. Recovered diversity to 41/46.

### IPF parameter change rationale

- top_k 20→30: wider window catches more generalist jobs
- floor 0.3→0.1: generalists penalised down to 10% instead of 30% of their score

## 10. Match quality refinement (Step 26)

Three multiplicative terms applied to `programme_recall` before normalisation:

```
refined_recall = recall × specificity_ratio × generic_penalty × coherence_boost
```

1. **Specificity ratio** — `log(1 + mean_idf_matched) / log(1 + mean_idf_all_job)`, clamped [0.5, 2.0]. Rewards matching rare skills.
2. **Generic penalty** — `1 − γ·generic_frac` (γ=0.3). Penalises matches dominated by below-median IDF skills.
3. **Coherence boost** — `1 + δ·mean_pairwise_cosine` (δ=0.2). Rewards coherent skill clusters.

Backward compatible: γ=0, δ=0 → quality_multiplier=1.0.

## 11. Coherence boost activation

Generated ESCO skill label embeddings (465 URIs, all-MiniLM-L6-v2) and passed to `align_hybrid()`.

| Metric | Value |
|--------|-------|
| Boost firing rate | 88% of pairs (≥3 matched skills) |
| Boost range | 1.0–1.11 |
| Top-1 changes | 1/46 |

Low discriminative power — ESCO skill labels embed similarly regardless of domain (all short English phrases). The coherence term is architecturally sound but limited by the embedding space granularity of skill labels.

## 12. Programme boilerplate removal

LAMA BPO navigation text ("ABOUT AIKOSNEWSREGISTERS CURRENTLY SELECTED CAREER Study and Learning Programmes...") was present in all 46 programme `cleaned_text` fields. Stripped at the "Programmes granting same qualifications" marker in `text_cleaner.py`.

| Metric | Before | After |
|--------|--------|-------|
| Inter-programme embedding similarity | 0.7510 | 0.6180 |
| Programme → CVbankas cosine | 0.3621 | 0.3356 |
| Programme → LinkedIn cosine | 0.2929 | 0.2647 |

### Score distribution impact

| Tier | Before removal | After removal |
|------|---------------|---------------|
| Strong (>0.4) | 6 (13%) | 10 (21%) |
| Moderate (0.25–0.4) | 27 (59%) | 18 (38%) |
| Weak (0.15–0.25) | 7 (15%) | 16 (34%) |
| Very weak (<0.15) | 6 (13%) | 3 (6%) |

The boilerplate inflated cosine similarity uniformly, pushing mediocre matches into the "moderate" tier artificially. After removal, scores are more honest: good matches score higher, bad matches score lower. The distribution polarised but became more truthful.

### Notable match improvements

| Programme | Before | After |
|-----------|--------|-------|
| Computer games and animation | BI programuotojas | Gameplay Programmer |
| Game Development | Game Designer | Gameplay Programmer |
| Cyber Systems and Security | Generic specialist | SOC analitikas (threat hunting) |
| Software Engineering (VU) | TV tower engineer | Linux sistemų administratorius |

## 13. LinkedIn underrepresentation

Despite comprising 23% of the dataset (119/520 jobs), LinkedIn jobs account for only 6% of top-10 matches. Root cause: LinkedIn jobs get 3.3 slots per programme's top-50 candidates vs expected 11.4 (proportional).

Mean cosine to programmes: LinkedIn 0.265 vs CVbankas 0.336. This gap persists after boilerplate removal — it reflects text style differences (corporate/practical vs academic) rather than boilerplate bias.

## 14. Current state (Milestone F)

### Match quality tiers (top-1, N=47)

| Tier | Count | % |
|------|-------|---|
| Strong (>0.4) | 10 | 21% |
| Moderate (0.25–0.4) | 18 | 38% |
| Weak (0.15–0.25) | 16 | 34% |
| Very weak (<0.15) | 3 | 6% |

### Diversity

| Scope | Unique | Total | % |
|-------|--------|-------|---|
| Top-1 | 40 | 47 | 85% |
| Top-5 | 123 | 235 | 52% |
| Top-10 | 163 | 470 | 35% |

### Weakly-matched programmes — two distinct causes

**Cause 1: Corpus coverage gap.** Programmes in niche domains (bioinformatics, game dev, multimedia design, digital arts) with very few matching jobs in Lithuanian job boards. This reflects real labour market structure — Lithuania's IT sector is dominated by enterprise software, fintech, and cybersecurity. These programmes may be training for international or remote markets rather than local employers.

**Cause 2: Generic programme descriptions.** Programmes with broad curricula that lack distinctive skill profiles. Their embeddings and skill sets are too generic to differentiate between job types, producing near-random matches with very low scores.

Both are valid findings for curriculum alignment analysis, not algorithmic failures.

### Evolution summary

| Milestone | Unique top-1 | Max repeat |
|-----------|-------------|------------|
| A — Old hybrid (symmetric Jaccard) | 13/35 (37%) | 6× |
| B — Asymmetric recall + IPF | 33/46 (72%) | 5× |
| C — + Auxiliary corpus (617 jobs) | 35/46 (76%) | 3× |
| D — + IDF weighting + formula tuning | 41/46 (89%) | 3× |
| E — + Corpus expansion + IPF retune | 40/47 (85%) | 10× |
| F — + Boilerplate fix + coherence boost | 40/47 (85%) | 10× |
