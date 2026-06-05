"""Pilot validation — Step 2: compare manual annotations vs extractor output.

Reads:
    experiments/validation/extraction/programmes_filled.csv
    experiments/validation/extraction/jobs_filled.csv
    data/dataset/dataset.parquet

Writes:
    experiments/validation/extraction/metrics.csv     — per-doc + micro avg
    experiments/validation/extraction/diff.csv        — TP/FP/FN/NEAR_MISS rows
    experiments/validation/extraction/REPORT.md       — defence-ready summary

Usage:
    python -m experiments.scripts.extraction_pilot_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.extraction_pilot import (
    DATASET_PATH,
    aggregate_metrics,
    compare_sample,
)
from src.scraping.config import DATA_DIR

OUTPUT_DIR = DATA_DIR.parent / "experiments" / "validation" / "extraction"


def _load_filled(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Filled annotation file {path} not found. "
            f"Copy programmes_template.csv → programmes_filled.csv (and the "
            f"jobs version), then fill in the esco_uri column."
        )
    return pd.read_csv(path).fillna("")


def _build_uri_to_label(dataset: pd.DataFrame) -> dict[str, str]:
    """Walk skill_details across all rows to harvest URI → preferred label."""
    table: dict[str, str] = {}
    for _, row in dataset.iterrows():
        details = row.get("skill_details", [])
        if details is None:
            continue
        if hasattr(details, "tolist"):
            details = details.tolist()
        for s in details:
            uri = s.get("esco_uri", "")
            if uri and uri not in table:
                table[uri] = s.get("preferred_label") or s.get("matched_text") or ""
    return table


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    df = df.fillna("")
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def _format_report(metrics: pd.DataFrame, diff: pd.DataFrame) -> str:
    lines: list[str] = ["# Extraction-pilot validation — report", ""]
    lines.append("## Per-document metrics")
    lines.append("")
    lines.append(_df_to_md(metrics))
    lines.append("")

    if not diff.empty:
        lines.append("## Error breakdown by document")
        lines.append("")
        grouped = diff.groupby(["doc_kind", "doc_id"], sort=False)
        for (kind, doc_id), block in grouped:
            counts = block["verdict"].value_counts().to_dict()
            counts_str = ", ".join(f"{k}={v}" for k, v in counts.items())
            lines.append(f"### {kind} {doc_id} ({counts_str})")
            lines.append("")
            lines.append(_df_to_md(block))
            lines.append("")

    return "\n".join(lines)


def main(output_dir: Path = OUTPUT_DIR) -> None:
    logger.info(f"Loading dataset from {DATASET_PATH}…")
    dataset = pd.read_parquet(DATASET_PATH)

    prog_path = output_dir / "programmes_filled.csv"
    jobs_path = output_dir / "jobs_filled.csv"
    progs_annot = _load_filled(prog_path)
    jobs_annot = _load_filled(jobs_path)

    annotations = pd.concat([progs_annot, jobs_annot], ignore_index=True)
    logger.info(
        f"Loaded {len(progs_annot)} programme rows + {len(jobs_annot)} job rows"
    )

    uri_to_label = _build_uri_to_label(dataset)
    cmps, diff = compare_sample(annotations, dataset, uri_to_label=uri_to_label)

    metrics = aggregate_metrics(cmps)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    diff_path = output_dir / "diff.csv"
    report_path = output_dir / "REPORT.md"

    metrics.to_csv(metrics_path, index=False)
    logger.info(f"Metrics → {metrics_path}")
    diff.to_csv(diff_path, index=False)
    logger.info(f"Diff → {diff_path}")
    report_path.write_text(_format_report(metrics, diff), encoding="utf-8")
    logger.info(f"Report → {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(output_dir=args.output_dir)
