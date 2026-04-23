# Implementation Progress

## Step 1 — Study Programme Data Collection [~]

Scrape LAMA BPO for Bachelor CS/ICT programmes (name, institution, mode, field, brief description).
Follow programme links to university websites for extended curriculum descriptions.
Exclude programmes with no descriptive text.

**Output:** `data/raw/programmes/`
**Module:** `src/scraping/lama_bpo.py`, `src/scraping/university_sites.py`

---

## Step 2 — Job Advertisement Data Collection [~]

Collect EU ICT/AI job postings from LinkedIn, TotalJobs, Upwork or similar.
Fields: title, description, required skills, employer sector, location, posting date.
Apply EU/Lithuania geographic + temporal filters.

**Output:** `data/raw/job_ads/`
**Module:** `src/scraping/job_ads.py`

---

## Step 3 — Text Preprocessing Pipeline [~]

Unified pipeline for both sources: language normalization (multilingual), text cleaning
(HTML, boilerplate, duplicates), tokenization. Reusable and documented.

**Output:** `data/processed/`
**Module:** `src/preprocessing/pipeline.py`

---

## Step 4 — Skill Extraction & Ontology Mapping [x]

Map explicit skills to ESCO ontology. Extract implicit skills via document embeddings
(Gugnani & Misra 2020). Produces symbolic representation per record.

**Output:** skill columns in processed dataset
**Module:** `src/skills/esco_mapper.py`, `src/skills/implicit_extractor.py`

---

## Step 4b — Skill Quality Filtering [x]

Post-process skill extraction output to remove noise before alignment:

1. Domain filtering — discard ESCO skills from irrelevant categories (e.g. sports, logistics, energy)
2. Frequency filtering — drop skills appearing in >80% of documents (uninformative)
3. Confidence threshold — raise minimum implicit skill confidence

Re-run steps 5–12 after filtering.

**Output:** overwrites `programmes_with_skills.parquet` and `jobs_with_skills.parquet`
**Module:** `src/skills/skill_filter.py`

---

## Step 5 — Semantic Embedding Generation [x]

Transformer-based dense embeddings for all programme descriptions (brief + extended)
and job postings. Stored alongside symbolic representations.

**Output:** embedding columns / separate embedding store
**Module:** `src/embeddings/generator.py`

---

## Step 6 — Dataset Assembly & Descriptive Validation [x]

Merge all data into single structured machine-readable dataset.
Compute descriptive stats (skill frequency, text length distributions, coverage rates).
Qualitative review of a representative subset.

**Output:** `data/dataset/`
**Module:** `src/dataset_builder.py`

---

## Step 7 — Clustering [x]

Cluster programme embeddings/skill vectors (specialization patterns).
Separately cluster job ads (labour-market demand groups).

**Output:** cluster labels in dataset
**Module:** `src/clustering/programme_clustering.py`, `src/clustering/job_clustering.py`

---

## Step 8 — Experiment 1: Skill-Based Symbolic Alignment [x]

Represent programmes + jobs as ESCO skill sets.
Compute overlap/weighted similarity. Produce ranked job list per programme.
Analyze skill gaps.

**Output:** `experiments/results/exp1_symbolic/`
**Module:** `src/alignment/symbolic.py`

---

## Step 9 — Experiment 2: Semantic Text-Based Alignment [x]

Cosine similarity and dot product between programme and job embeddings.
Both metrics computed on the same embedding matrix; results stored as separate
columns in rankings.parquet for direct comparison in Step 11.
Run twice: once with `embedding` (combined), once with `embedding_brief` vs
`embedding_extended` to measure effect of curricular detail.

**Output:** `experiments/results/exp2_semantic/`
**Module:** `src/alignment/semantic.py`

---

## Step 10 — Experiment 3: Hybrid Alignment [x]

Embedding-based retrieval refined by skill-based overlap.
Balances recall (semantic) with transparency (symbolic).

**Output:** `experiments/results/exp3_hybrid/`
**Module:** `src/alignment/hybrid.py`

---

## Step 11 — Cross-Strategy Evaluation [x]

Compare all 3 approaches: ranking consistency, stability, strategy agreement.
Domain expert evaluation for meaningful skill overlap confirmation.

**Output:** `experiments/results/evaluation/`
**Module:** `src/evaluation/cross_strategy.py`

---

## Step 12 — Recommendations [x]

Actionable curriculum enhancement recommendations:
skill gaps, emerging market trends, best alignment approach for ongoing monitoring.

**Output:** `experiments/results/recommendations/`
**Module:** `src/recommendations/generator.py`

---

## Step 13 — End-to-End Pipeline [x]

Orchestrates all steps (1–12) in sequence.
Skips completed steps unless --force is set.
Supports --from N and --steps N,N,... flags.

**Output:** all step outputs in order
**Module:** `src/pipeline.py`

---

## Step 14 — Bug Fixes & Data Integrity [x]

Fix `skills_per_record` metric in `dataset_builder.py` — currently reports 0 for all 345 records.
Root cause: parquet stores lists as numpy ndarray, `isinstance(x, list)` misses them.

Note: `embedding_brief` is all zeros because the LAMA BPO source has no brief descriptions — not a bug.

**Output:** corrected `data/dataset/stats.json`
**Module:** `src/dataset_builder.py`

---

## Step 15 — Hybrid Alpha Sensitivity Analysis [x]

Sweep alpha ∈ [0.0, 0.1, 0.2, ... 1.0] for hybrid alignment.
For each alpha, compute Spearman correlation with symbolic/semantic, Jaccard@10, and hybrid score distribution.
Produce alpha sensitivity curve.

**Output:** `experiments/results/sensitivity/alpha_sweep.parquet`, `alpha_sweep_summary.json`
**Module:** `src/evaluation/sensitivity.py`

---

## Step 16 — Statistical Significance Testing [x]

1. Bootstrap confidence intervals (1000 resamples over 46 programmes) on Spearman correlations
2. Wilcoxon signed-rank test on paired per-programme scores to test if strategy differences are significant
3. Effect sizes (rank-biserial correlation)

**Output:** `experiments/results/evaluation/significance.json`
**Module:** `src/evaluation/significance.py`

---

## Step 17 — Consensus-Based IR Metrics [x]

Use cross-strategy agreement as proxy relevance: jobs in top-K of ≥2 strategies = "relevant".
Compute Precision@K, NDCG@K, MRR, Coverage@K for each strategy against consensus set.

**Output:** `experiments/results/evaluation/ir_metrics.json`
**Module:** `src/evaluation/ir_metrics.py`

---

## Step 18 — Cluster-Stratified Alignment Analysis [x]

1. Programme-cluster × job-cluster contingency table + chi-squared test
2. Per-cluster alignment score distributions (all 3 strategies)
3. Cluster-specific skill gaps — which specializations have the largest market mismatch
4. Strategy performance by cluster — does symbolic/semantic/hybrid vary by specialization

**Output:** `experiments/results/evaluation/cluster_analysis.json`, `cluster_analysis.parquet`
**Module:** `src/evaluation/cluster_analysis.py`

---

## Step 19 — BM25 Baseline [x]

Add TF-IDF/BM25 text retrieval baseline over cleaned_text.
Rank job ads per programme by BM25 score. Include in cross-strategy evaluation as reference.

**Output:** `experiments/results/exp0_bm25/`
**Module:** `src/alignment/bm25_baseline.py`

---

## Step 20 — Extraction Ablation Study [x]

Remove S1/S2/S3/S4 modules one at a time from explicit extraction.
Re-run symbolic alignment with each ablated skill set.
Measure impact on weighted Jaccard and skill gap coverage.

**Output:** `experiments/results/ablation/`
**Module:** `src/evaluation/ablation.py`

---

## Step 21 — Bootstrap Ranking Stability [x]

Resample 80% of job ads 100 times. For each resample, re-run all 3 alignment strategies.
Measure rank stability (Kendall tau between full and resampled rankings per programme).

**Output:** `experiments/results/evaluation/stability.json`
**Module:** `src/evaluation/stability.py`

---

## Step 22 — Expanded Job Corpus (LinkedIn) [x]

Scrape additional IT/CS job ads from LinkedIn to expand the job corpus beyond CVbankas.
Industry filters: SOFTWARE_DEVELOPMENT, TECHNOLOGY_INTERNET, IT_SERVICES.
Location: Lithuania. Merged with CVbankas jobs, deduplicated by title.

**Result:** 122 LinkedIn + 275 CVbankas = 397 unique jobs (was 299).
LinkedIn jobs are richer: mean 19.2 skills vs 13.6 for CVbankas, higher implicit ratio (44.4% vs 39.4%).
Added 43 new ESCO URIs, 8 of which match programme skills.

**Output:** `data/raw/job_ads/linkedin_jobs.json`, merged `all_jobs.json`
**Module:** `src/scraping/linkedin.py` (new), `src/scraping/job_ads.py` (merge logic)

---

## Step 22b — Generalist Job Penalty & Hybrid Redesign [x]

Address generalist job descriptions dominating top rankings across many programmes.

Changes applied:

1. **Asymmetric programme_recall** replaced symmetric weighted Jaccard — measures fraction of job-demanded skill weight the programme covers
2. **Per-programme min-max normalisation** — cosine and recall normalised to [0,1] within each programme's candidate set
3. **Inverse Programme Frequency (IPF)** — `log(1 + N_prog / count_top_k(j))` with floor=0.3, penalises generalist jobs
4. **Auxiliary corpus** — 617 EU-wide LinkedIn jobs for implicit skill extractor fitting (not alignment)

**Results:** Unique top-1 jobs: 13/35 → 35/46. Score CoV: 0.12 → 0.50. Top-5 generalist jobs (freq>5): 13 → 5.

**Output:** `experiments/results/exp3_hybrid/FINDINGS.md`
**Module:** `src/alignment/hybrid.py`, `src/scraping/linkedin_auxiliary.py`, `src/skills/skill_mapper.py`

---

## Step 23 — IDF + ESCO Reuse-Level Skill Weighting [x]

Replace uniform skill weights in symbolic alignment with a two-factor weighting scheme:

1. **ESCO `reuseLevel` tier:** transversal=0.3, cross-sector=0.5, sector-specific=0.8, occupation-specific=1.0
2. **Corpus IDF factor:** multiply tier weight by `log(N / df(uri))` where N=total docs, df=docs containing URI

Apply to `_build_weighted_skills` in symbolic alignment: final weight = tier_weight × idf_factor × (1.0 if explicit, 0.5 if implicit).
Re-run symbolic + hybrid alignment and compare Jaccard/overlap distributions with uniform-weight baseline.

**Rationale:** Current symbolic alignment treats all ESCO URIs equally. Generic skills ("communication", "teamwork") contribute the same overlap as specialised ones ("Kubernetes", "NLP"). This dilutes the signal — programmes and jobs share many generic skills, inflating Jaccard for poor matches and compressing the score range (mean Jaccard = 0.062). Weighting by specificity and corpus rarity should widen the score distribution and make top-ranked matches more meaningful.

**Output:** `experiments/results/exp1_symbolic_weighted/`
**Module:** `src/alignment/symbolic.py` (extended), `src/skills/skill_weights.py` (new)

---

## Step 24 — Hybrid Formula Tuning [x]

Systematic comparison of hybrid scoring formula variants across 5 aspects (α sweep 0.0–1.0, step=0.05):

1. **Normalisation** — minmax vs rank-based. Min-max wins (diversity 0.89 vs 0.85).
2. **Agreement boost** (β=0.0–0.3) — hurts diversity and increases generalists. Discarded.
3. **Combination function** — linear vs geometric vs harmonic. Linear wins (0.89 vs 0.83/0.87). Discarded.
4. **IPF parameters** — swept ipf_top_k ∈ {0,5,10,15,20}, ipf_floor ∈ {0.1,0.3,0.5}. **k=20, floor=0.3 applied** (same diversity, generalists 6→3, max_freq 11→7).
5. **Candidate pool** — swept semantic_top_n ∈ {20,30,50,75,100}. top_n=50 kept (sufficient).

Applied changes:

- α: 0.50 → **0.60** (more weight to semantic signal)
- top_k: 10 → **20** (wider generalist penalty window)

Not applied (no improvement):

- Rank normalisation, agreement boost, geometric/harmonic combination, semantic_top_n change, IPF floor change

**Result:** Top-1 diversity 35/46 (0.76) → 41/46 (0.89). Top-5 generalists: 5→3. Max top-5 freq: 13→7.

**Output:** `experiments/results/sensitivity/formula_variants.json`
**Module:** `src/evaluation/formula_tuning.py` (new), `src/alignment/hybrid.py` (updated defaults)

---

## Step 26 — Match Quality Refinement [x]

Refine `programme_recall` before normalisation with three multiplicative terms:

1. **Specificity ratio** — `log(1 + mean_idf_matched) / log(1 + mean_idf_all_job)`, clamped [0.5, 2.0]
2. **Generic penalty** — `1 - γ·generic_frac` where generic_frac = IDF weight of below-median matched URIs
3. **Coherence boost** — `1 + δ·mean_pairwise_cosine` over matched skill embeddings (≥3 skills)

Backward compatible: γ=0, δ=0 → quality_multiplier=1.0 → identical to previous formula.

**Output:** integrated into `src/alignment/hybrid.py`
**Module:** `src/alignment/hybrid.py` (new `compute_match_quality()`), `src/skills/skill_weights.py` (new `compute_median_idf()`)

---

## Step 25 — Larger Embedding Model [x]

Compared `all-MiniLM-L6-v2` (384-dim, 22M params) against `all-mpnet-base-v2` (768-dim, 109M params).
Re-generated embeddings for all 46 programmes and 390 job ads with MPNet.
Re-ran semantic + hybrid alignment and compared against MiniLM baseline.

**Result — MiniLM retained.** MPNet's +5-point STS benchmark advantage did not translate to better alignment:

- Semantic CoV: 0.300 (MiniLM) vs 0.207 (MPNet) — MiniLM scores are more discriminative
- Hybrid top-1 diversity: 41/46 (MiniLM) vs 39/46 (MPNet) — MiniLM produces more diverse matches
- Top-5 generalists: 3 (MiniLM) vs 7 (MPNet) — MPNet reintroduces generalist dominance
- Cross-model top-1 agreement: 1/46 (2%) — almost entirely different rankings
- Cross-model Spearman (top-20): 0.05 (semantic), 0.22 (hybrid) — near-random overlap

**Root cause:** Both models share a 256-token truncation limit (~1000 chars). 96% of programmes exceed this. Swapping the model doesn't help because the same information is discarded.

**Fix applied:** Section-weighted programme embeddings (parse into subjects/outcomes/identity/specialisations, embed each independently, weighted average) and chunk-and-pool job embeddings (split into 256-token chunks, mean-pool). Also removed 2 VGTU programmes with insufficient descriptions (24 and 571 chars).

**Output:** `experiments/results/evaluation/embedding_comparison.json`
**Module:** `src/embeddings/generator.py`, `src/preprocessing/pipeline.py`

---

## Step 27 — ESCO Description Embeddings for Coherence Boost [x]

Replace ESCO label embeddings (2-3 word labels) with ESCO skill description embeddings (1-3 sentences) in coherence boost computation.
Current coherence boost fires in 88% of pairs but only ranges 1.0–1.11 due to coarse label embeddings.
Description embeddings should produce meaningful pairwise cosine similarity between matched skills.

Added `build_skill_description_embeddings()` and `save_skill_embeddings()` to `skill_weights.py`.
Embeds the ESCO `description` field (1-3 sentences) instead of short labels. Saves to `data/dataset/skill_embeddings.npz`.
The existing `_load_skill_embeddings()` in `hybrid.py` loads these for coherence boost.

**Output:** updated `src/skills/skill_weights.py`
**Module:** `src/skills/skill_weights.py` (ESCO description embedding builder + NPZ persistence)
**Tests:** 9 tests in `tests/skills/test_skill_embeddings.py`

---

## Step 28 — Two-Tier IPF [x]

Replace single IPF floor with two-tier penalty:

- Jobs appearing in top-K of >50% of programmes → strict floor (0.05)
- Other popular jobs → standard floor (0.1)
  Preserves fallback matches for niche programmes while harder-penalising universal generalists.

**Output:** updated `src/alignment/hybrid.py`
**Module:** `src/alignment/hybrid.py`

---

## Step 29 — Confidence-Aware Normalisation [x]

When all candidates for a programme have similar raw scores, min-max stretches small differences into full [0,1] range, making rankings fragile.
Add dampening factor: if raw score range (max-min) is below a threshold, shrink the normalised range proportionally.
Prevents noisy rankings for programmes with uniformly weak matches.

**Output:** updated `src/alignment/hybrid.py`
**Module:** `src/alignment/hybrid.py`

---

## Step 30 — LinkedIn Boilerplate Stripping [x]

Strip corporate boilerplate from LinkedIn job descriptions: "About the job" header, benefit/offer sections, EEO blocks, salary lines, and data protection notices. Uses cutoff approach — first matching non-technical section truncates remaining text.

57% of LinkedIn jobs affected, 18.4% total char reduction. Zero CVbankas false positives. Top-1 diversity dropped 39→35 due to stripped embeddings being more focused, but max hybrid score increased 0.59→0.71.

**Output:** updated `src/preprocessing/text_cleaner.py`, `src/preprocessing/pipeline.py`
**Module:** `src/preprocessing/text_cleaner.py` (`strip_linkedin_boilerplate()`), `tests/preprocessing/test_text_cleaner.py`

---

## Step 31 — Programme-Level Skill TF-IDF [x]

Weight each programme's skills by distinctiveness relative to other programmes (inter-programme IDF), not just corpus-wide IDF.
A skill unique to 1 programme should matter more in matching than one shared by 20 programmes.

Added `compute_programme_idf(df)` to `skill_weights.py` — filters to programme rows only and computes IDF.
Added `use_programme_idf` parameter to `align_symbolic_weighted()` — when True, programme skills use inter-programme IDF while job skills keep corpus-wide IDF. Default False for backward compatibility.

**Output:** updated `src/skills/skill_weights.py`, `src/alignment/symbolic.py`
**Module:** `src/skills/skill_weights.py` (`compute_programme_idf`), `src/alignment/symbolic.py` (`use_programme_idf` param)
**Tests:** 5 tests in `tests/skills/test_skill_embeddings.py`, 4 tests in `tests/alignment/test_symbolic.py`

---

## Step 32 — Niche Domain Coverage Analysis [x]

Analyse per-programme job coverage to identify niche domains with insufficient matches.
Flag low-coverage programmes (< min_matches above score threshold) and generate corpus expansion recommendations.

Added `src/evaluation/coverage.py` with:
- `analyse_coverage()` — per-programme coverage metrics (n_matches, coverage_ratio, top_score, low_coverage flag)
- `identify_niche_clusters()` — aggregate coverage by cluster to find niche domain groups
- `generate_expansion_recommendations()` — actionable recommendations for low-coverage programmes with top skill URIs

**Output:** `experiments/results/coverage/programme_coverage.parquet`, `niche_clusters.parquet`, `coverage_summary.json`
**Module:** `src/evaluation/coverage.py`
**Tests:** 12 tests in `tests/evaluation/test_coverage.py`

---

## Step 33 — Impact Comparison & Coherence Boost Removal [x]

Evaluated Steps 27 and 31 against a baseline (no skill embeddings, no programme IDF) in four configurations: baseline, +desc_emb only, +prog_idf only, +both.

- **Programme IDF (Step 31):** Clearly positive — 60% of programmes improved, +0.018 mean score lift, top-1 diversity 35→37. Generic roles replaced by domain-specific ones.
- **ESCO description embeddings (Step 27):** Actively harmful — 53% degraded, -0.002 mean delta. Coherence boost added noise.

Applied: enabled `use_programme_idf=True` as default, removed coherence boost (`delta`, `skill_embeddings`, `_load_skill_embeddings`, `build_skill_description_embeddings`, `save_skill_embeddings`). Re-ran pipeline steps 8+10.

**Output:** `experiments/results/impact_comparison/`
**Module:** `src/evaluation/impact_comparison.py`, `src/alignment/hybrid.py` (simplified)
**Tests:** 478 tests, all passing (11 coherence tests removed)

---

## Step 34 — Expand Section Header Recognition [ ]

The section-weighted embedding strategy (Step 25) misses 84 of 102 unique headers in programme descriptions. Only 18 are mapped to the four section groups (subjects, outcomes, identity, specialisations). Unmapped content falls into `_remainder` which gets **zero weight** — averaging 2,625 chars per programme (27% of text).

This means learning outcomes like "students will develop special skills", "graduates will be able to", and specialisation blocks like "Specialization – Machine Learning" are silently discarded from the embedding.

Changes:

1. Expand `_SECTION_MAP` in `generator.py` to cover the 84 unmapped headers, mapping each to the correct group
2. Add a `_remainder` fallback weight (e.g. 0.05) so truly unmapped content contributes minimally rather than being zeroed
3. 15/45 programmes have no recognised "subjects" header and 14/45 have no "outcomes" — after header expansion these should drop significantly
4. Re-generate embeddings and re-run hybrid alignment

**Rationale:** 58% of programme text by volume is currently identity+remainder (weighted 0.20 + 0.00 = 0.20), while discriminative content (subjects+outcomes+specialisations) averages only 42% of text but gets 80% of weight. Expanding header recognition will route more discriminative content into the high-weight groups, increasing embedding discrimination.

**Output:** updated embeddings in `data/embeddings/`, re-run alignment
**Module:** `src/embeddings/generator.py`

---

## Step 35 — Rebalance Hybrid Alpha [ ]

Cosine similarity has very low discriminative power: per-programme range across 50 candidates is only 0.154 (std=0.034). Programme recall has 2x the range (0.306) and 1.7x the std (0.056). Despite this, cosine gets 60% of the weight (alpha=0.6).

The hybrid top-1 matches the semantic-only top-1 in only 2% of cases, vs 20% for recall-only top-1. This means the IPF penalty — not the alpha-weighted combination — is the dominant re-ranker.

Changes:

1. Re-run alpha sweep at finer granularity (0.3–0.7, step=0.025) with programme IDF enabled
2. Measure top-1 diversity, score mean/CoV, and Spearman correlation with pure semantic/symbolic rankings at each alpha
3. Evaluate whether Step 34 (header expansion) sufficiently improves cosine discrimination — if it does, alpha may not need lowering
4. Apply optimal alpha

**Rationale:** The current alpha=0.6 was tuned before programme IDF was enabled. With programme IDF amplifying the recall signal, the balance point may have shifted. Additionally, if cosine ranges remain narrow after Step 34, lowering alpha to 0.4–0.5 would let the more discriminative recall component drive rankings.

**Output:** `experiments/results/sensitivity/`
**Module:** `src/evaluation/sensitivity.py`

---

## Legend

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
