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

## Step 34 — Expand Section Header Recognition [x]

Expanded `_SECTION_MAP` from 18 to 90 entries. Added `_remainder` group with weight 0.05. Rebalanced weights: subjects=0.35, outcomes=0.25, identity=0.15, specialisations=0.20, _remainder=0.05.

Results: programmes with 0 subjects dropped 15→1, with 0 outcomes 14→7. Top-1 diversity improved 37→39/45 (0.867). Inter-programme cosine increased 0.633→0.697 (programmes become more similar when more shared curriculum is captured). Cosine range per programme barely changed (0.154→0.160). Net score impact mixed (38% improved, 59% degraded, mean -0.007) — expected, proper fix is Step 35 alpha rebalance.
4. Re-generate embeddings and re-run hybrid alignment

**Rationale:** 58% of programme text by volume is currently identity+remainder (weighted 0.20 + 0.00 = 0.20), while discriminative content (subjects+outcomes+specialisations) averages only 42% of text but gets 80% of weight. Expanding header recognition will route more discriminative content into the high-weight groups, increasing embedding discrimination.

**Output:** updated embeddings in `data/embeddings/`, re-run alignment
**Module:** `src/embeddings/generator.py`

---

## Step 35 — Rebalance Hybrid Alpha [x]

Fine-grained alpha sweep (0.3–0.7, step=0.025) with full hybrid pipeline (programme IDF, match quality, confidence-aware normalisation, two-tier IPF).

Three alphas achieve max diversity (40/45 = 0.889): 0.300, 0.550, 0.575. α=0.300 rejected (70% recall = not truly hybrid). α=0.55 selected: smallest shift from 0.6, maintains hybrid character, improves diversity 39→40/45, generalists 2→1.

**Applied:** alpha 0.6 → 0.55

**Output:** `experiments/results/sensitivity/alpha_rebalance.json`
**Module:** `src/evaluation/sensitivity.py` (`run_alpha_rebalance()`), `src/alignment/hybrid.py` (default updated)

---

## Step 36 — Replace LinkedIn Data Source (Commercial Viability) [ ]

**Deferred — not required for thesis defence. Only relevant if project is productised.**

LinkedIn ToS explicitly prohibits scraping; *hiQ v. LinkedIn* (2022) settled in LinkedIn's favour. Commercial use of the existing LinkedIn-derived corpus is legally exposed (ToS breach + GDPR risk on personal data in postings). Research/thesis use is a tolerated grey area and does not require action now.

Replace LinkedIn ingestion with licensed / open sources before any paid pilot or commercial release:

1. **EURES** (EU Employment Services API) — free, official, EU-wide coverage, pristine provenance
2. **Adzuna API** — licensed commercial aggregator, ~€500–€2k/mo, strong UK/DE/FR/NL/PL
3. **CVbankas.lt partnership** — negotiate commercial licence for existing LT coverage
4. **Optional fallbacks:** Jooble / Careerjet partner feeds, Cedefop Skills-OVATE, SerpAPI / JobsPikr (legal risk outsourced by contract)

Rebuild `all_jobs.json` without LinkedIn rows, re-run steps 3–11, re-validate ranking quality against the existing expert-reviewed baseline. Dropping LinkedIn likely improves procurement story ("fully licensed data, auditable provenance, GDPR-compliant") — not a loss for the commercial pitch.

**Rationale:** keeps the LinkedIn-inclusive pipeline intact for the thesis; isolates the replacement as a commercialisation concern to be handled only when needed.

**Output:** replacement job corpus in `data/raw/job_ads/` (licensed sources only), re-run evaluation results
**Module:** new `src/scraping/eures.py`, new `src/scraping/adzuna.py`, updated `src/scraping/job_ads.py`

---

## Step 37 — Confidence-Weighted Implicit Skills (T2a) [x]

Replace the blanket `0.5` weight currently assigned to every implicit skill with a confidence-proportional weight using the propagation cosine that is already stored in `skill_details.confidence` (range 0.70–1.00 after Step 4b filtering).

**Result — sqrt selected and applied as default.** Three modes (uniform / linear / sqrt) compared on the full hybrid pipeline. All three preserve top-1 diversity (40/45). Sqrt reduces head-tied programmes (top-1↔top-2 gap < 0.02) from 15 → 12 (-20%) while keeping the top-1 max score at 0.677. Linear is too aggressive — drops max score 7% and removes only 2 tied heads. Top-1 score-mean cost of sqrt is 0.005 (negligible). Default `implicit_confidence_mode` flipped from `"uniform"` to `"sqrt"` across `build_weighted_skills`, `align_symbolic_weighted`, and `align_hybrid`. Full results: `experiments/results/exp_implicit_confidence/FINDINGS.md`.

Proposed mapping:

```
w_implicit(u, d) = 0.5 × clip((conf - 0.70) / 0.30, 0, 1)   # → 0 at conf=0.70, 0.5 at conf=1.00
```

(Try a linear and a square-root variant; pick by impact on hybrid metrics.)

**Why:** ~47% of programme skill weight and ~35% of job skill weight comes from the implicit channel, but the propagation confidence is currently discarded at the alignment boundary. A skill propagated at cosine 0.95 carries genuinely stronger evidence than one at 0.71. The information is already in the dataset.

**Risk:** lowers total recall mass uniformly — re-tune γ if the generic_penalty saturates. Cheap fix, low blast radius.

**Output:** updated `src/skills/skill_weights.py`, comparison report
**Module:** `src/skills/skill_weights.py` (`build_weighted_skills`), `src/alignment/symbolic.py` (callers)
**Metrics to report:** top-1 diversity, top-1 score mean / max, gap top1↔top2 distribution, sym↔hyb Spearman.

---

## Step 38 — Cross-Encoder Re-ranking of Top-N Candidates (T1a) [x]

Implemented the re-ranking stage as `src/alignment/cross_encoder.py` (with `score_pairs` and `score_pairs_sectioned`) plus `xe_alpha` / `xe_pool_mode` parameters on `align_hybrid`. Tested two re-ranker models (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-base`) across single-pass + section-weighted + section-max pooling and three blend configurations. Section-aware variants reuse Step 34's section parser and weights.

**Result — partial improvement, not adopted as default.** With MS MARCO-MiniLM, three-channel single-pass cuts head-tied programmes (gap<0.02) 12→5 at the cost of −2 unique top-1 and +1 generalist; section-weighted pooling improves diversity 40→43 and removes top-5 generalists 1→0 but loses the head-tie win. Per-programme top-1 quality is roughly even with baseline (10 better / 5 wash / 8 worse changes for secwm); aggregate diversity wins come from spreading picks. bge-reranker-base is more conservative (Spearman base↔hyb 0.91 vs 0.89), regresses head discrimination (gap<0.02 12→14), fixes only 1.5/4 persistent niche regressions. Default hybrid stays at α=0.55, no cross-encoder.

Persistent regressions (Game Designer → QA Tester, SOC analyst → Test Manager, AI Engineer → SysAdmin, Gameplay Programmer → Senior Java) trace to **specificity asymmetry**: long generic IT job descriptions over-cover broad curricula on lexical surface vs short specialised descriptions. Re-ranker model upgrades cannot fix this — belongs to Step 39 (LTR).

**Output:** `experiments/results/exp3_hybrid_xenc/` (FINDINGS.md, summary.json, rankings_*.parquet, top1_diff.csv), `experiments/results/exp3_hybrid_xenc_bge/`
**Module:** `src/alignment/cross_encoder.py` (new), `src/alignment/hybrid.py` (xe_alpha + xe_pool_mode), `src/evaluation/cross_encoder_experiment.py` (new), `tests/conftest.py` (MockCrossEncoder), `tests/alignment/test_cross_encoder.py` (23 tests)

---

## Step 39 — Learning-to-Rank with Cross-Strategy Consensus (T1b) [x] tried, not adopted — implementation removed

Prototyped LightGBM `LGBMRanker` (LambdaRank) over the per-programme top-50 semantic candidate pool (45 programmes × 50 = 2,250 pairs), trained with leave-one-programme-out CV. Features (13): `cosine_score`, `programme_recall`, `weighted_jaccard`, `overlap_coeff`, `bm25_score`, `n_matched_uris`, `mean_idf_matched`, `mean_idf_job_unmatched`, `count_top_k` (IPF input), `prog_skill_richness`, `job_skill_count`, `prog_implicit_ratio`, `job_implicit_ratio`. Consensus labels intentionally excluded hybrid (sources: symbolic ∪ semantic ∪ BM25, ≥ 2 of 3) so the learner was not trained against the formula it was meant to replace.

**Result — not adopted; implementation removed.** Held-out IR metrics on consensus looked strong (NDCG@10 = 0.759, P@5 = 0.324, MRR = 0.750, Coverage@10 = 0.973 over 37/45 evaluable programmes). The number is misleading on its own. Two diagnostics killed adoption:

1. Gain-based feature importance was dominated by `count_top_k` (1456) and `job_skill_count` (493) — both popularity proxies — almost 2× the next feature `weighted_jaccard` (729).
2. Mean per-programme Spearman against the hybrid baseline was **−0.332** (anti-correlated, not just different).

Cross-strategy consensus surfaces jobs all three strategies pick precisely because long, generic IT job descriptions over-cover everything — the same specificity-asymmetry effect Step 38 traced for cross-encoders. The learner took the popularity shortcut and rediscovered the generalist bias that hybrid's IPF was designed to suppress. Adopting LTR would have undone Step 22b's diversity gains. NDCG-against-consensus is not a sufficient proxy for human relevance in this corpus.

Implementation deleted to keep `main` lean: `src/alignment/ltr.py`, `tests/alignment/test_ltr.py`, the regenerable artefacts under `experiments/results/exp4_ltr/`, and the `lightgbm` dependency are gone. **`experiments/results/exp4_ltr/FINDINGS.md` is retained as the experiment record** with full numbers, feature-importance table, and follow-up directions (`min_strategies=3`, drop popularity features, post-hoc IPF rescaling, hand-labelled spot-check before any future LTR attempt). Default rankings stay at hybrid (α=0.55, no cross-encoder).

**Output:** `experiments/results/exp4_ltr/FINDINGS.md` (retained)
**Code in tree:** none

---

## Step 40 — Asymmetric Retrieval Encoder (T1c) [ ]

Replace `all-MiniLM-L6-v2` for programme-vs-job retrieval with an encoder explicitly trained on asymmetric query/document retrieval: `intfloat/e5-large-v2` or `BAAI/bge-large-en-v1.5`. Use prefixed inputs (`query: {programme}` and `passage: {job}`) so the asymmetry between curriculum descriptions and recruitment text is modelled directly.

**Why deferred to last:** Step 25 already showed MPNet (a larger symmetric encoder) gives no improvement because both models share the 256-token truncation limit. Asymmetric encoders are a *different* change — they're trained with explicit query/document distinction and have higher MTEB retrieval scores — but they cost ~30 min to re-embed both corpora and re-run the full pipeline. Run only after Steps 37–39 land; if those bring head discrimination into a strong regime, the embedding upgrade may be unnecessary.

Section-weighted programme embeddings (Step 34) and chunk-and-pool job embeddings already exist and must be preserved when swapping the encoder.

**Risk:** a 1024-dim embedding instead of 384-dim — must verify storage (`embedding`, `embedding_brief`, `embedding_extended`, `skill_embeddings.npz`) handles the new dimension everywhere. Model size ~1.3 GB instead of 80 MB.

**Output:** `experiments/results/evaluation/asymmetric_encoder/`, side-by-side hybrid metrics
**Module:** `src/embeddings/generator.py` (model parameter + prefix handling), regenerated embeddings
**Metrics to report:** programme-job cosine mean / range, top-1 diversity, top-1 mean score, head-discrimination gap, sensitivity to model choice.

---

## Legend

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
