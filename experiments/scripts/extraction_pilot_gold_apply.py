"""Materialise the refined gold from gold_curation.csv decisions.

Reads the `accepted_action` column from gold_curation.csv and writes:
  experiments/validation/extraction/programmes_filled_v2.csv
  experiments/validation/extraction/jobs_filled_v2.csv

Only rows where `accepted_action == 'keep'` are included. The output schema
matches the original `*_filled.csv` files so downstream attribution scripts
can use the v2 gold by pointing at the new files.

Then re-runs the verifier attribution against the refined gold for each
suffix supplied, so we can see what the cleanup does to F1.

Usage:
  python -m experiments.scripts.extraction_pilot_gold_apply \\
      --variants qwen2.5-3B_b1 qwen2.5-3B_v2_title
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.scraping.config import DATA_DIR

VALIDATION_DIR = DATA_DIR.parent / "experiments" / "validation" / "extraction"

ORIGINAL_COLUMNS = [
    "doc_kind", "doc_id", "doc_title", "esco_uri", "preferred_label",
    "annotation_type", "annotator_confidence", "notes",
]


def apply_decisions(validation_dir: Path) -> tuple[int, int]:
    curation = pd.read_csv(validation_dir / "gold_curation.csv")

    # Recover the original `annotator_confidence` column (and any others not in
    # curation) by joining against the original `*_filled.csv` files on (doc_kind, esco_uri).
    progs_orig = pd.read_csv(validation_dir / "programmes_filled.csv").fillna("")
    jobs_orig = pd.read_csv(validation_dir / "jobs_filled.csv").fillna("")
    orig = pd.concat([progs_orig, jobs_orig], ignore_index=True)[
        ["doc_kind", "doc_id", "esco_uri", "annotator_confidence"]
    ]
    curation = curation.merge(orig, on=["doc_kind", "doc_id", "esco_uri"], how="left")
    curation["annotator_confidence"] = curation["annotator_confidence"].fillna("high")

    missing = curation["accepted_action"].isna() | (curation["accepted_action"].astype(str).str.strip() == "")
    if missing.any():
        logger.warning(f"{missing.sum()} rows have no accepted_action — they will be EXCLUDED")

    kept = curation[curation["accepted_action"] == "keep"].copy()
    dropped = curation[curation["accepted_action"] == "drop"].copy()
    logger.info(f"Decisions: keep={len(kept)}, drop={len(dropped)}, missing={int(missing.sum())}")

    progs_v2 = kept[kept["doc_kind"] == "programme"][ORIGINAL_COLUMNS].copy()
    jobs_v2 = kept[kept["doc_kind"] == "job_ad"][ORIGINAL_COLUMNS].copy()

    progs_path = validation_dir / "programmes_filled_v2.csv"
    jobs_path = validation_dir / "jobs_filled_v2.csv"
    progs_v2.to_csv(progs_path, index=False)
    jobs_v2.to_csv(jobs_path, index=False)
    logger.info(f"Wrote {progs_path}  ({len(progs_v2)} rows)")
    logger.info(f"Wrote {jobs_path}   ({len(jobs_v2)} rows)")
    return len(progs_v2), len(jobs_v2)


def rescore_against_v2(suffix: str, validation_dir: Path) -> dict:
    """Run the same attribution as extraction_pilot_attribution.py but against the v2 gold."""
    sfx = f"_{suffix}" if suffix else ""

    # Load v2 gold
    progs = pd.read_csv(validation_dir / "programmes_filled_v2.csv").fillna("")
    jobs = pd.read_csv(validation_dir / "jobs_filled_v2.csv").fillna("")
    gold = pd.concat([progs, jobs], ignore_index=True)
    gold = gold[gold["esco_uri"].str.strip().ne("")]
    gold["esco_uri"] = gold["esco_uri"].str.strip()
    gold["doc_id"] = gold["doc_id"].astype(int)

    # Decisions
    dec = pd.read_csv(validation_dir / f"llm_verify_decisions{sfx}.csv")
    dec["esco_uri"] = dec["esco_uri"].astype(str).str.strip()
    dec["doc_id"] = dec["doc_id"].astype(int)
    kept = dec[dec["kept"].astype(bool)][["doc_kind", "doc_id", "esco_uri"]].drop_duplicates()
    kept["_kept"] = 1
    extracted = dec[["doc_kind", "doc_id", "esco_uri"]].drop_duplicates()
    extracted["_extracted"] = 1

    gold["_g"] = 1
    keys = ["doc_kind", "doc_id", "esco_uri"]
    merged = (
        gold.merge(extracted, on=keys, how="outer")
        .merge(kept, on=keys, how="left")
        .fillna({"_g": 0, "_extracted": 0, "_kept": 0})
    )
    for c in ("_g", "_extracted", "_kept"):
        merged[c] = merged[c].astype(int)

    tp = int(((merged["_g"] == 1) & (merged["_kept"] == 1)).sum())
    fp = int(((merged["_g"] == 0) & (merged["_kept"] == 1)).sum())
    fn = int(((merged["_g"] == 1) & (merged["_kept"] == 0)).sum())
    verifier_drop = int(
        ((merged["_g"] == 1) & (merged["_kept"] == 0) & (merged["_extracted"] == 1)).sum()
    )
    extractor_miss = int(
        ((merged["_g"] == 1) & (merged["_extracted"] == 0)).sum()
    )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ceiling = (tp + verifier_drop) / (tp + fn) if (tp + fn) else 0.0
    return {
        "suffix": suffix,
        "gold": int(merged["_g"].sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "verifier_drop": verifier_drop,
        "extractor_miss": extractor_miss,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "extractor_recall_ceiling": round(ceiling, 3),
    }


def main(variants: list[str], validation_dir: Path = VALIDATION_DIR) -> None:
    n_progs, n_jobs = apply_decisions(validation_dir)

    rows: list[dict] = []
    for variant in variants:
        try:
            rows.append(rescore_against_v2(variant, validation_dir))
        except FileNotFoundError as e:
            logger.warning(f"  skip {variant}: {e}")
    if rows:
        df = pd.DataFrame(rows)
        path = validation_dir / "GOLD_V2_RESCORE.csv"
        df.to_csv(path, index=False)
        print()
        print(df.to_string(index=False))
        print()
        print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["qwen2.5-3B_b1", "qwen2.5-3B_v2_title"])
    parser.add_argument("--validation-dir", type=Path, default=VALIDATION_DIR)
    args = parser.parse_args()
    main(variants=args.variants, validation_dir=args.validation_dir)
