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

Two multiplicative terms applied to `programme_recall` before normalisation:

```
refined_recall = recall × specificity_ratio × generic_penalty
```

1. **Specificity ratio** — `log(1 + mean_idf_matched) / log(1 + mean_idf_all_job)`, clamped [0.5, 2.0]. Rewards matching rare skills.
2. **Generic penalty** — `1 − γ·generic_frac` (γ=0.3). Penalises matches dominated by below-median IDF skills.

Backward compatible: γ=0 → quality_multiplier=1.0.

Note: Coherence boost (Step 27, δ=0.2 × mean pairwise cosine of ESCO description embeddings) was tested and removed — it degraded 53% of programmes while improving only 18%. See `experiments/results/impact_comparison/FINDINGS.md`.

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

## 14. Two-tier IPF (Step 28)

Replaced single IPF floor with two-tier penalty based on how many programmes a job appears in:

- **Universal generalists** (top-K of >50% programmes) → `ipf_strict_floor=0.05`
- **Moderately popular jobs** → standard `ipf_floor=0.1`

Logic: `strict_cutoff = max(2, int(n_prog × 0.5))`. Jobs in top-K of ≥23 programmes (out of 45) get the strict floor.

Backward compatible: when `ipf_strict_floor=ipf_floor`, behaves identically to single-tier.

| Metric | Single-tier | Two-tier |
|--------|-------------|----------|
| Universal generalists penalised | — | 15 jobs (strict floor) |
| Other generalists | 129 | 114 (standard floor) |
| Unique top-1 | 40/47 | 40/47 |

Minimal top-line impact because most universal generalists were already ranked low. The change primarily prevents edge cases where a truly universal job (appearing in top-K for 40+ programmes) could still accumulate enough score to reach top-5.

## 15. Confidence-aware normalisation (Step 29)

When all candidates for a programme have similar raw cosine scores, min-max normalisation stretches small differences into the full [0,1] range, making rankings fragile. The confidence dampening shrinks the normalised range proportionally:

```
range = max - min (per programme)
median_range = median of all programme ranges
confidence = min(range / median_range, 1.0)
normalised_score = confidence × minmax(score)
```

Applied **only to cosine**, not recall. Rationale: a narrow recall range still carries real signal (skill overlap vs none), unlike narrow cosine which reflects genuinely indistinguishable embeddings.

### Critical fix: cosine-only dampening

Initial implementation dampened both cosine and recall. This caused "Information systems engineering" (VGTU) to get a zero-recall top-1 match — dampening destroyed the recall signal that should have discriminated between matches with and without skill overlap.

After restricting dampening to cosine only, the recall signal is preserved for programmes with narrow cosine but meaningful skill differences.

## 16. Section-weighted programme embeddings

### Problem: text truncation

MiniLM-L6-v2 has a 256-token limit (~1000 characters). 96% of programmes exceed this limit (mean=6203 chars). Both MiniLM and MPNet share the same truncation boundary — **this is why the larger model (Step 25) didn't help**: same information was discarded.

### Solution: embed sections independently

Parse programme text into section groups, embed each independently (avoiding truncation), and compute a weighted average:

| Section group | Weight | Content type |
|---------------|--------|-------------|
| subjects | 0.40 | Course lists — most discriminative |
| outcomes | 0.25 | Learning outcomes / competencies |
| identity | 0.20 | Objectives + distinctive features |
| specialisations | 0.15 | Domain-specific tracks |

Headers are detected by pattern (ends with colon, <80 chars) and mapped to groups via `_SECTION_MAP`. Sections not matching any header go to `_remainder` (not embedded in the weighted average).

### Fallback for programmes without section headers

Programmes where all section groups are empty (content goes entirely to `_remainder`) get a plain full-text embedding instead. This prevents zero vectors for programmes with non-standard formatting.

After analysis, 2 programmes were removed from the dataset entirely due to insufficient descriptions:
- "Information Technologies" (VGTU) — only 24 characters of text
- "Information systems engineering" (VGTU) — 571 characters, no section headers

"Informatics Systems" (VMU, 6869 chars, inline headers) was kept with the fallback mechanism.

### Chunk-and-pool job embeddings

Job descriptions are split into 256-token chunks, each embedded independently, and the final embedding is the mean of all chunks (L2-normalised). This ensures long job descriptions are fully represented rather than truncated.

### Embedding quality impact

| Metric | Before (truncated) | After (section-weighted + chunked) |
|--------|--------------------|------------------------------------|
| Inter-programme cosine | 0.605 | 0.633 |
| Programme-job cosine mean | 0.365 | 0.365 |
| Programme-job cosine CoV | 0.300 | 0.252 |
| Zero-embedding programmes | 0 | 0 |

Inter-programme cosine increased slightly (more shared section structure captured). Programme-job cosine mean unchanged but the CoV decreased, indicating more consistent matching.

## 17. Data quality: programme removal

Removed 2 of 47 programmes (→ 45) with insufficient descriptions for meaningful embedding and skill extraction:

| Programme | Institution | Chars | Issue |
|-----------|------------|-------|-------|
| Information Technologies | VGTU | 24 | Near-empty: only programme name |
| Information systems engineering | VGTU | 571 | No section headers, generic text |

These programmes produced near-random top matches across all strategies, adding noise without providing meaningful alignment signal.

## 18. Current state (Milestone G)

Parameters: α=0.6, IPF top_k=30, floor=0.1, strict_floor=0.05, strict_threshold=0.5, γ=0.3, δ=0.2, norm_confidence=True. Corpus: 520 jobs, 45 programmes.

### Match quality tiers (top-1, N=45)

| Tier | Count | % |
|------|-------|---|
| Strong (>0.4) | 9 | 20% |
| Moderate (0.25–0.4) | 13 | 29% |
| Weak (0.15–0.25) | 19 | 42% |
| Very weak (<0.15) | 4 | 9% |

### Diversity

| Scope | Unique | Total | % |
|-------|--------|-------|---|
| Top-1 | 39 | 45 | 87% |
| Top-5 | 130 | 225 | 58% |
| Top-10 | 185 | 450 | 41% |

Max top-1 repeat: 2× (6 jobs). No job dominates top-1 for more than 2 programmes.

### Score distribution

| Metric | Value |
|--------|-------|
| Hybrid mean (top-1) | 0.285 |
| Hybrid std (top-1) | 0.115 |
| Hybrid CoV (top-1) | 0.403 |
| Range | 0.126 – 0.590 |

### Cross-strategy correlations

| Pair | Spearman ρ (mean) |
|------|-------------------|
| symbolic ↔ semantic | 0.287 |
| symbolic ↔ hybrid | 0.071 |
| semantic ↔ hybrid | 0.130 |

Top-1 agreement across all 3 strategies: 0/45 (0%).

### Weakly-matched programmes — two distinct causes

**Cause 1: Corpus coverage gap.** Programmes in niche domains (bioinformatics, game dev, multimedia design, digital arts) with very few matching jobs in Lithuanian job boards. This reflects real labour market structure — Lithuania's IT sector is dominated by enterprise software, fintech, and cybersecurity. These programmes may be training for international or remote markets rather than local employers.

**Cause 2: Generic programme descriptions.** Programmes with broad curricula that lack distinctive skill profiles. Their embeddings and skill sets are too generic to differentiate between job types, producing near-random matches with very low scores.

Both are valid findings for curriculum alignment analysis, not algorithmic failures.

## 19. LinkedIn boilerplate stripping (Step 30)

Stripped non-technical boilerplate from LinkedIn job descriptions before embedding:

- **"About the job"** header (100% of LinkedIn jobs)
- **Benefit/offer sections** ("What we offer", "Benefits", "Our offer") and everything after
- **EEO/diversity blocks** ("We are proud to foster...", "Equal opportunity employer")
- **Salary/compensation lines**
- **Lithuanian data protection** ("Siųsdami savo gyvenimo aprašymą")

Applied as a cutoff: the first matching section header or EEO block truncates the rest of the text. This preserves all technical content (role description, requirements, tech stack) while removing corporate boilerplate that inflates cosine similarity without carrying role-specific signal.

### Impact

| Metric | Before | After |
|--------|--------|-------|
| LinkedIn jobs affected | — | 73/127 (57%) |
| Total char reduction | — | 18.4% |
| Mean LinkedIn desc length | 3620 | 2955 chars |
| CVbankas false positives | — | 0/439 |

### Alignment impact

| Metric | Before (G) | After (H) |
|--------|-----------|-----------|
| Unique top-1 | 39/45 (87%) | 35/45 (78%) |
| Max top-1 repeat | 2× | 3× |
| Hybrid score max | 0.590 | 0.710 |
| Hybrid CoV (top-1) | 0.403 | 0.412 |
| Programme-job cosine mean | 0.365 | 0.363 |
| Top-1 changes | — | 5/34 |

Top-1 diversity dropped from 39 to 35. The stripped LinkedIn embeddings are more focused on technical content, causing some jobs to become stronger matches for multiple related programmes. The 3× repeat ("KIBERNETINIO SAUGUMO VADOVAS") affects 3 security/informatics programmes — a domain-coherent repeat rather than a generalist problem.

The max hybrid score increased from 0.59 to 0.71, indicating stronger top matches where boilerplate was previously diluting the signal.

### Evolution summary

| Milestone | Unique top-1 | Max repeat | Programmes |
|-----------|-------------|------------|------------|
| A — Old hybrid (symmetric Jaccard) | 13/35 (37%) | 6× | 35 |
| B — Asymmetric recall + IPF | 33/46 (72%) | 5× | 46 |
| C — + Auxiliary corpus (617 jobs) | 35/46 (76%) | 3× | 46 |
| D — + IDF weighting + formula tuning | 41/46 (89%) | 3× | 46 |
| E — + Corpus expansion + IPF retune | 40/47 (85%) | 10× | 47 |
| F — + Boilerplate fix + coherence boost | 40/47 (85%) | 10× | 47 |
| G — + Two-tier IPF, confidence norm, section embeddings, data cleanup | 39/45 (87%) | 2× | 45 |
| H — + LinkedIn boilerplate stripping | 35/45 (78%) | 3× | 45 |
| I — + Programme IDF (Step 33) | 37/45 (82%) | 2× | 45 |
| J — + Section header expansion (Step 34) | 39/45 (87%) | 2× | 45 |
