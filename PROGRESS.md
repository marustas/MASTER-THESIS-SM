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

## Step 24 — Finer Alpha Sweep [ ]
Re-run hybrid alpha sensitivity analysis with step=0.01 (101 alpha values) instead of current step=0.1 (11 values).
Focus on the region around the current optimum (±0.15) at step=0.005 for high-resolution curve.
Report optimal alpha with bootstrap 95% CI.

**Rationale:** The current 0.1 step sweep identifies the best alpha only within ±0.05 precision. The hybrid score is `α·cosine + (1-α)·jaccard`, and the sensitivity curve may have a narrow peak — especially after score normalisation or skill reweighting changes the Jaccard distribution. A finer sweep ensures the reported optimal alpha is not an artefact of coarse discretisation, and the bootstrap CI quantifies how stable this optimum is across programme subsets.

**Output:** `experiments/results/sensitivity/alpha_sweep_fine.parquet`, `alpha_sweep_fine_summary.json`
**Module:** `src/evaluation/sensitivity.py` (extended)

---

## Step 25 — Larger Embedding Model [ ]
Replace `all-MiniLM-L6-v2` (384-dim, 22M params) with `all-mpnet-base-v2` (768-dim, 109M params).
Re-generate embeddings for all programmes and job ads.
Re-run semantic + hybrid alignment and compare cosine score distributions, Spearman correlations, and IR metrics against the MiniLM baseline.

**Rationale:** `all-MiniLM-L6-v2` is optimised for speed over accuracy. On the STS benchmark it scores 0.788 Spearman, while `all-mpnet-base-v2` scores 0.838 — a 5-point gap. For a thesis with only 46×299 pairs, inference speed is irrelevant but embedding quality directly affects semantic alignment accuracy. A stronger model should produce more discriminative cosine scores (current mean=0.366, max=0.684), separating truly relevant matches from noise.

**Output:** `data/processed/*/embeddings_mpnet.parquet`, `experiments/results/exp2_semantic_mpnet/`
**Module:** `src/embeddings/generator.py` (parameterised model name)

---

## Legend
- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
