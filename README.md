# Study Program Content Mapping to Automatically Mined Market Demands

Master's thesis — Vilnius Gediminas Technical University
AI Engineering MSc | Stanislau Marudau
Supervisor: Prof. Dr. Simona Ramanauskaitė

## Overview

This project automates the estimation of alignment between EU university study programmes and labour market demands using NLP and semantic similarity techniques.

## Project Structure

```
├── src/
│   ├── scraping/       # Step 1-2: LAMA BPO + job ad scrapers
│   ├── preprocessing/  # Step 3: unified text preprocessing pipeline
│   ├── skills/         # Step 4: ESCO mapping + implicit skill extraction
│   ├── embeddings/     # Step 5: transformer embedding generation
│   ├── clustering/     # Step 7: programme + job ad clustering
│   ├── alignment/      # Step 8-10: symbolic, semantic, hybrid alignment
│   └── evaluation/     # Step 11: cross-strategy evaluation
├── data/
│   ├── raw/            # scraped raw data (not committed)
│   ├── processed/      # cleaned + preprocessed data (not committed)
│   └── dataset/        # final structured dataset
├── notebooks/          # EDA, experiments, visualizations
├── experiments/        # configs and results
└── tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in credentials
```

## Implementation Progress

See [PROGRESS.md](PROGRESS.md) for detailed step-by-step status.
