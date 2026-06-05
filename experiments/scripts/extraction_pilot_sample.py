"""Pilot validation — Step 1: sample documents and write annotation templates.

Run once.  Produces two CSVs under ``experiments/validation/extraction/``:

    programmes_template.csv  — 5 programmes × 8 blank rows = 40 rows
    jobs_template.csv        — 5 jobs × 8 blank rows = 40 rows

Plus a ``selection.json`` recording the seed and the picked doc IDs for
reproducibility, and a ``DOCS_TO_READ.md`` cheat-sheet with the full
text of the sampled documents so the annotator does not need to open
the parquet by hand.

Usage:
    python -m experiments.scripts.extraction_pilot_sample
    python -m experiments.scripts.extraction_pilot_sample --n 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.extraction_pilot import (
    DATASET_PATH,
    stratified_sample_jobs,
    stratified_sample_programmes,
    write_annotation_template,
)
from src.scraping.config import DATA_DIR

OUTPUT_DIR = DATA_DIR.parent / "experiments" / "validation" / "extraction"


def _write_docs_to_read(
    progs: pd.DataFrame, jobs: pd.DataFrame, path: Path
) -> None:
    """Markdown cheat-sheet with the full text of each sampled document."""
    lines: list[str] = ["# Documents to read — extraction pilot", ""]

    lines.append("## Programmes")
    lines.append("")
    for _, row in progs.iterrows():
        lines.append(f"### Programme {row['programme_id']} — {row['name']}")
        lines.append("")
        lines.append(f"- Institution: {row.get('institution', '')}")
        lines.append(f"- Cluster: {row.get('cluster_label', '')}")
        lines.append("")
        lines.append("```")
        lines.append(str(row.get("extended_description", "")).strip())
        lines.append("```")
        lines.append("")

    lines.append("## Job ads")
    lines.append("")
    for _, row in jobs.iterrows():
        lines.append(f"### Job {row['job_id']} — {row['job_title']}")
        lines.append("")
        lines.append(f"- Sector: {row.get('employer_sector', '')}")
        lines.append(f"- Cluster: {row.get('cluster_label', '')}")
        lines.append("")
        lines.append("```")
        lines.append(str(row.get("description", "")).strip())
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Docs cheat-sheet → {path}")


def main(n: int = 5, seed: int = 42, output_dir: Path = OUTPUT_DIR) -> None:
    logger.info(f"Loading dataset from {DATASET_PATH}…")
    dataset = pd.read_parquet(DATASET_PATH)

    progs = stratified_sample_programmes(dataset, n=n, seed=seed)
    jobs = stratified_sample_jobs(dataset, n=n, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    write_annotation_template(
        progs, output_dir / "programmes_template.csv", doc_kind="programme"
    )
    write_annotation_template(
        jobs, output_dir / "jobs_template.csv", doc_kind="job_ad"
    )

    _write_docs_to_read(progs, jobs, output_dir / "DOCS_TO_READ.md")

    selection = {
        "seed": seed,
        "n_per_side": n,
        "programmes": [
            {"programme_id": int(r["programme_id"]),
             "name": r["name"],
             "cluster_label": float(r["cluster_label"])}
            for _, r in progs.iterrows()
        ],
        "jobs": [
            {"job_id": int(r["job_id"]),
             "job_title": r["job_title"],
             "cluster_label": float(r["cluster_label"])}
            for _, r in jobs.iterrows()
        ],
    }
    with open(output_dir / "selection.json", "w") as fh:
        json.dump(selection, fh, indent=2, ensure_ascii=False)
    logger.info(f"Selection record → {output_dir / 'selection.json'}")

    logger.info(
        f"Done. Open {output_dir}/DOCS_TO_READ.md and fill in "
        f"programmes_template.csv / jobs_template.csv blind, then run "
        f"experiments.scripts.extraction_pilot_compare."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(n=args.n, seed=args.seed, output_dir=args.output_dir)
