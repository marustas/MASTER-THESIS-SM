"""Pilot threshold sweep — re-run implicit extraction at lower thresholds.

Tests whether lowering the implicit cosine threshold (currently 0.70 in the
canonical pipeline) improves the pilot's implicit-stream recall against the
gold standard.

Procedure:
  1. Load dataset.parquet (texts + canonical skill_details).
  2. Build the corpus skill sets from the canonical explicit skills.
  3. Fit ImplicitSkillExtractor on the corpus once (real model, real embeddings).
  4. For each candidate threshold τ ∈ {0.50, 0.60, 0.70 (baseline)}:
       a. Set extractor._sim_threshold = τ and re-extract implicit skills for
          the 10 pilot documents.
       b. Build a modified dataset where the 10 pilot docs' skill_details are
          replaced with (canonical explicit) + (newly-extracted implicit
          filtered at confidence ≥ τ).
       c. Run compare_sample against the same annotation CSVs.
       d. Aggregate metrics.

Writes:
  experiments/validation/extraction/threshold_sweep_metrics.csv
  experiments/validation/extraction/THRESHOLD_SWEEP_REPORT.md

Usage:
  python -m experiments.scripts.extraction_pilot_threshold_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.extraction_pilot import (
    DATASET_PATH,
    _coerce_details,
    aggregate_metrics,
    compare_sample,
)
from src.scraping.config import DATA_DIR
from src.skills.explicit_extractor import ExtractedSkill, ExplicitSkillExtractor
from src.skills.implicit_extractor import ImplicitSkillExtractor

OUTPUT_DIR = DATA_DIR.parent / "experiments" / "validation" / "extraction"

PILOT_PROG_IDS = [12, 21, 25, 26, 29]
PILOT_JOB_IDS = [1, 20, 193, 253, 429]


def _details_to_extracted_skills(details: list[dict]) -> list[ExtractedSkill]:
    """Convert dataset's skill_details dicts back into ExtractedSkill objects."""
    out: list[ExtractedSkill] = []
    for d in details:
        out.append(ExtractedSkill(
            esco_uri=d.get("esco_uri", ""),
            preferred_label=d.get("preferred_label", ""),
            matched_text=d.get("matched_text", ""),
            explicit=bool(d.get("explicit", False)),
            implicit=bool(d.get("implicit", False)),
            confidence=float(d.get("confidence", 1.0)),
        ))
    return out


def _skill_to_dict(s: ExtractedSkill) -> dict:
    return {
        "esco_uri": s.esco_uri,
        "preferred_label": s.preferred_label,
        "matched_text": s.matched_text,
        "explicit": s.explicit,
        "implicit": s.implicit,
        "confidence": s.confidence,
    }


def _build_corpus(dataset: pd.DataFrame) -> tuple[list[str], list[list[ExtractedSkill]]]:
    """Pull cleaned_text + canonical explicit skills for the implicit extractor."""
    texts: list[str] = []
    explicit_skill_sets: list[list[ExtractedSkill]] = []
    for _, row in dataset.iterrows():
        text = row.get("cleaned_text") or row.get("description") or row.get("extended_description") or ""
        details = _coerce_details(row.get("skill_details"))
        all_skills = _details_to_extracted_skills(details)
        explicit_only = [s for s in all_skills if s.explicit]
        texts.append(str(text))
        explicit_skill_sets.append(explicit_only)
    return texts, explicit_skill_sets


def _replace_pilot_details(
    dataset: pd.DataFrame,
    new_implicit: dict[tuple[str, int], list[ExtractedSkill]],
    threshold: float,
) -> pd.DataFrame:
    """Return a copy of dataset with the 10 pilot rows' skill_details replaced.

    For each pilot doc, the new skill_details = (canonical explicit kept) +
    (newly-extracted implicit filtered by confidence ≥ threshold).
    """
    df = dataset.copy()
    # Recompute the positional index used by the pilot sampler
    progs = df[df["source_type"] == "programme"].reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    jobs = df[df["source_type"] == "job_ad"].reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    for pid in PILOT_PROG_IDS:
        orig_idx = int(progs.iloc[pid]["_orig_idx"])
        details = _coerce_details(df.at[orig_idx, "skill_details"])
        explicit_dicts = [d for d in details if d.get("explicit")]
        new_imps = new_implicit.get(("programme", pid), [])
        new_imp_dicts = [
            _skill_to_dict(s) for s in new_imps if s.confidence >= threshold
        ]
        df.at[orig_idx, "skill_details"] = explicit_dicts + new_imp_dicts

    for jid in PILOT_JOB_IDS:
        orig_idx = int(jobs.iloc[jid]["_orig_idx"])
        details = _coerce_details(df.at[orig_idx, "skill_details"])
        explicit_dicts = [d for d in details if d.get("explicit")]
        new_imps = new_implicit.get(("job_ad", jid), [])
        new_imp_dicts = [
            _skill_to_dict(s) for s in new_imps if s.confidence >= threshold
        ]
        df.at[orig_idx, "skill_details"] = explicit_dicts + new_imp_dicts

    return df


def _extract_pilot_implicit(
    extractor: ImplicitSkillExtractor,
    dataset: pd.DataFrame,
) -> dict[tuple[str, int], list[ExtractedSkill]]:
    """Run extract() for each of the 10 pilot docs, return implicit-skill lists."""
    progs = dataset[dataset["source_type"] == "programme"].reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    jobs = dataset[dataset["source_type"] == "job_ad"].reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    out: dict[tuple[str, int], list[ExtractedSkill]] = {}

    for pid in PILOT_PROG_IDS:
        orig_idx = int(progs.iloc[pid]["_orig_idx"])
        row = dataset.loc[orig_idx]
        text = row.get("cleaned_text") or row.get("extended_description") or ""
        details = _coerce_details(row.get("skill_details"))
        explicit_uris = {d["esco_uri"] for d in details if d.get("explicit") and d.get("esco_uri")}
        out[("programme", pid)] = extractor.extract(
            text=str(text), explicit_uris=explicit_uris, doc_idx=orig_idx,
        )

    for jid in PILOT_JOB_IDS:
        orig_idx = int(jobs.iloc[jid]["_orig_idx"])
        row = dataset.loc[orig_idx]
        text = row.get("cleaned_text") or row.get("description") or ""
        details = _coerce_details(row.get("skill_details"))
        explicit_uris = {d["esco_uri"] for d in details if d.get("explicit") and d.get("esco_uri")}
        out[("job_ad", jid)] = extractor.extract(
            text=str(text), explicit_uris=explicit_uris, doc_idx=orig_idx,
        )

    return out


def main(
    output_dir: Path = OUTPUT_DIR,
    thresholds: tuple[float, ...] = (0.50, 0.60, 0.70),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {DATASET_PATH}…")
    dataset = pd.read_parquet(DATASET_PATH)

    logger.info("Loading annotations…")
    annotations = pd.concat(
        [
            pd.read_csv(output_dir / "programmes_filled.csv").fillna(""),
            pd.read_csv(output_dir / "jobs_filled.csv").fillna(""),
        ],
        ignore_index=True,
    )

    logger.info("Building corpus index…")
    corpus_texts, corpus_explicit = _build_corpus(dataset)
    logger.info(f"  {len(corpus_texts)} documents in corpus")

    logger.info("Fitting ImplicitSkillExtractor (this loads the embedding model)…")
    dummy_explicit_extractor = ExplicitSkillExtractor.__new__(ExplicitSkillExtractor)
    extractor = ImplicitSkillExtractor(
        explicit_extractor=dummy_explicit_extractor,
        sim_threshold=min(thresholds),  # set to lowest, override per loop
        top_k=10,
    )
    extractor.fit(texts=corpus_texts, explicit_skills_per_doc=corpus_explicit)

    sweep_rows: list[dict] = []
    per_doc_per_threshold: list[dict] = []

    for tau in thresholds:
        logger.info(f"--- Threshold τ = {tau} ---")
        extractor._sim_threshold = tau
        new_implicit = _extract_pilot_implicit(extractor, dataset)

        # Log how many implicit URIs per doc
        for (kind, doc_id), skills in new_implicit.items():
            kept = [s for s in skills if s.confidence >= tau]
            logger.info(f"  {kind} #{doc_id}: {len(skills)} raw implicit, {len(kept)} above τ")

        modified_ds = _replace_pilot_details(dataset, new_implicit, threshold=tau)
        cmps, diff = compare_sample(annotations, modified_ds)
        agg = aggregate_metrics(cmps)
        micro = agg[agg["doc_kind"] == "ALL"].iloc[0]

        sweep_rows.append({
            "threshold": tau,
            "tp": int(micro["tp"]),
            "fp": int(micro["fp"]),
            "fn": int(micro["fn"]),
            "near_miss": int(micro["near_miss"]),
            "precision": float(micro["precision"]),
            "recall": float(micro["recall"]),
            "f1": float(micro["f1"]),
        })

        for _, row in agg.iterrows():
            if row["doc_kind"] == "ALL":
                continue
            per_doc_per_threshold.append({
                "threshold": tau,
                "doc_kind": row["doc_kind"],
                "doc_id": int(row["doc_id"]),
                "doc_title": row["doc_title"],
                "gold_n": int(row["gold_n"]),
                "extracted_n": int(row["extracted_n"]),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            })

    sweep_df = pd.DataFrame(sweep_rows)
    per_doc_df = pd.DataFrame(per_doc_per_threshold)

    sweep_path = output_dir / "threshold_sweep_metrics.csv"
    per_doc_path = output_dir / "threshold_sweep_per_doc.csv"
    sweep_df.to_csv(sweep_path, index=False)
    per_doc_df.to_csv(per_doc_path, index=False)
    logger.info(f"Sweep metrics → {sweep_path}")
    logger.info(f"Per-doc breakdown → {per_doc_path}")

    # Report
    lines: list[str] = ["# Implicit-extractor threshold sweep — pilot results", ""]
    lines.append("Micro-averaged metrics on the 10 pilot documents at each tested threshold:")
    lines.append("")
    lines.append("| τ | TP | FP | FN | NM | Precision | Recall | F1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sweep_rows:
        lines.append(
            f"| {r['threshold']:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['near_miss']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    report_path = output_dir / "THRESHOLD_SWEEP_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Report → {report_path}")

    logger.info("Sweep complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.60, 0.70],
        help="Cosine thresholds to test (default: 0.50 0.60 0.70)",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, thresholds=tuple(args.thresholds))
