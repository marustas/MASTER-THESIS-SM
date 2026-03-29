# Study Program Content Mapping to Automatically Mined Market Demands

Master's thesis — Vilnius Gediminas Technical University
AI Engineering MSc | Stanislau Marudau
Supervisor: Prof. Dr. Simona Ramanauskaitė

## Overview

Automates alignment estimation between EU university study programmes and labour
market demands using ESCO skill ontology, transformer embeddings, and three
complementary alignment strategies (symbolic, semantic, hybrid).

## Requirements

- Python 3.11.14 (managed via pyenv — `.python-version` is committed)
- Playwright browsers (for scraping)
- ESCO v1.2.1 taxonomy CSV (for skill extraction)
- Scraping credentials in `.env` (see `.env.example`)

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Install Playwright browsers (needed for scraping only)
playwright install chromium

# 4. Install spaCy language model
python -m spacy download en_core_web_sm

# 5. Copy and fill in scraping credentials
cp .env.example .env

# 6. Place ESCO taxonomy CSV
# Download from https://esco.ec.europa.eu/en/use-esco/download
# Version=v1.2.1 | Content=classification | Language=en | Format=csv
# Save to: data/raw/esco/skills_en.csv
```

## Running the pipeline

### Full pipeline (all steps)

```bash
python -m src.pipeline
```

### Start from a specific step

```bash
python -m src.pipeline --from 8
```

### Run specific steps only

```bash
python -m src.pipeline --steps 3,4,5
```

### Force re-run (ignore existing outputs)

```bash
python -m src.pipeline --force
```

### Run individual steps

```bash
python -m src.scraping.lama_bpo          # Step 1 — scrape study programmes
python -m src.scraping.job_ads           # Step 2 — scrape job ads
python -m src.preprocessing.pipeline    # Step 3 — text preprocessing
python -m src.skills.skill_mapper       # Step 4 — ESCO skill extraction
python -m src.embeddings.generator      # Step 5 — transformer embeddings
python -m src.dataset_builder           # Step 6 — dataset assembly
python -m src.clustering.programme_clustering  # Step 7a — programme clusters
python -m src.clustering.job_clustering        # Step 7b — job ad clusters
python -m src.alignment.symbolic        # Step 8 — symbolic alignment
python -m src.alignment.semantic        # Step 9 — semantic alignment
python -m src.alignment.hybrid          # Step 10 — hybrid alignment
python -m src.evaluation.cross_strategy # Step 11 — cross-strategy evaluation
python -m src.recommendations.generator # Step 12 — curriculum recommendations
```

## Tests

All tests run fully offline (~9s). No network access or model downloads required.

```bash
# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run a single test file
.venv/bin/python -m pytest tests/alignment/test_symbolic.py -v

# Run a single test
.venv/bin/python -m pytest tests/alignment/test_symbolic.py::TestWeightedJaccard::test_identical_sets -v
```

## Lint

```bash
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m ruff format src/ tests/
```

## Project structure

```
src/
  scraping/           Steps 1–2: LAMA BPO programme + job ad scrapers
  preprocessing/      Step 3:    text cleaning, deduplication, language detection
  skills/             Step 4:    ESCO skill extraction (4-module ensemble)
  embeddings/         Step 5:    sentence-transformer embeddings (all-MiniLM-L6-v2)
  dataset_builder.py  Step 6:    dataset assembly and descriptive validation
  clustering/         Step 7:    K-Means / HDBSCAN clustering by source type
  alignment/          Steps 8–10: symbolic, semantic, and hybrid alignment
  evaluation/         Step 11:   cross-strategy Spearman + Jaccard evaluation
  recommendations/    Step 12:   curriculum gap analysis and job match ranking
  pipeline.py         Step 13:   end-to-end orchestrator

data/
  raw/                scraped source data (not committed)
  processed/          preprocessed + skill-enriched parquets (not committed)
  embeddings/         embedding parquets (not committed)
  dataset/            unified dataset.parquet + stats.json (not committed)

experiments/
  results/
    exp1_symbolic/    symbolic alignment output
    exp2_semantic/    semantic alignment output
    exp3_hybrid/      hybrid alignment output
    evaluation/       cross-strategy metrics
    recommendations/  final curriculum recommendations

tests/                offline test suite (212 tests)
```

## Pipeline outputs

| Step | Output |
|------|--------|
| 1 | `data/raw/programmes/lama_bpo_programmes.json` |
| 2 | `data/raw/job_ads/all_jobs.json` |
| 3 | `data/processed/*/preprocessed.parquet` |
| 4 | `data/processed/*/with_skills.parquet` |
| 5 | `data/embeddings/*.parquet` |
| 6 | `data/dataset/dataset.parquet` |
| 7 | `cluster_label` column added to `dataset.parquet` |
| 8 | `experiments/results/exp1_symbolic/` |
| 9 | `experiments/results/exp2_semantic/` |
| 10 | `experiments/results/exp3_hybrid/` |
| 11 | `experiments/results/evaluation/` |
| 12 | `experiments/results/recommendations/` |
