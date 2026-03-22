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

## Step 4 — Skill Extraction & Ontology Mapping [ ]
Map explicit skills to ESCO ontology. Extract implicit skills via document embeddings
(Gugnani & Misra 2020). Produces symbolic representation per record.

**Output:** skill columns in processed dataset
**Module:** `src/skills/esco_mapper.py`, `src/skills/implicit_extractor.py`

---

## Step 5 — Semantic Embedding Generation [ ]
Transformer-based dense embeddings for all programme descriptions (brief + extended)
and job postings. Stored alongside symbolic representations.

**Output:** embedding columns / separate embedding store
**Module:** `src/embeddings/generator.py`

---

## Step 6 — Dataset Assembly & Descriptive Validation [ ]
Merge all data into single structured machine-readable dataset.
Compute descriptive stats (skill frequency, text length distributions, coverage rates).
Qualitative review of a representative subset.

**Output:** `data/dataset/`
**Module:** `src/dataset_builder.py`, `notebooks/06_descriptive_validation.ipynb`

---

## Step 7 — Clustering [ ]
Cluster programme embeddings/skill vectors (specialization patterns).
Separately cluster job ads (labour-market demand groups).

**Output:** cluster labels in dataset
**Module:** `src/clustering/programme_clustering.py`, `src/clustering/job_clustering.py`

---

## Step 8 — Experiment 1: Skill-Based Symbolic Alignment [ ]
Represent programmes + jobs as ESCO skill sets.
Compute overlap/weighted similarity. Produce ranked job list per programme.
Analyze skill gaps.

**Output:** `experiments/results/exp1_symbolic/`
**Module:** `src/alignment/symbolic.py`

---

## Step 9 — Experiment 2: Semantic Text-Based Alignment [ ]
Cosine similarity between programme and job embeddings.
Ranked job lists per programme.
Compare brief vs. extended descriptions for alignment accuracy.

**Output:** `experiments/results/exp2_semantic/`
**Module:** `src/alignment/semantic.py`

---

## Step 10 — Experiment 3: Hybrid Alignment [ ]
Embedding-based retrieval refined by skill-based overlap.
Balances recall (semantic) with transparency (symbolic).

**Output:** `experiments/results/exp3_hybrid/`
**Module:** `src/alignment/hybrid.py`

---

## Step 11 — Cross-Strategy Evaluation [ ]
Compare all 3 approaches: ranking consistency, stability, strategy agreement.
Domain expert evaluation for meaningful skill overlap confirmation.

**Output:** `experiments/results/evaluation/`
**Module:** `src/evaluation/cross_strategy.py`

---

## Step 12 — Recommendations [ ]
Actionable curriculum enhancement recommendations:
skill gaps, emerging market trends, best alignment approach for ongoing monitoring.

**Output:** `notebooks/12_recommendations.ipynb`, final report section

---

## Legend
- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
