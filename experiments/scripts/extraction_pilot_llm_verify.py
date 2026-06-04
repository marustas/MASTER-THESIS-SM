"""Pilot — apply local-LLM verification on the canonical extractor output.

For each of the 10 pilot documents, take its canonical skill_details
(explicit + implicit), ask a small instruct-tuned LLM whether each
extracted URI is genuinely taught/required by the document, and keep
only those the verifier confirms.  Compare against the same gold
standard.

Default model: Qwen/Qwen2.5-1.5B-Instruct (~3 GB, MPS-friendly on
Apple Silicon).  First run downloads the model into the HuggingFace
cache.

Writes:
  experiments/validation/extraction/llm_verify_metrics.csv
  experiments/validation/extraction/llm_verify_per_doc.csv
  experiments/validation/extraction/llm_verify_decisions.csv
  experiments/validation/extraction/LLM_VERIFY_REPORT.md

Usage:
  python -m experiments.scripts.extraction_pilot_llm_verify
  python -m experiments.scripts.extraction_pilot_llm_verify --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import time
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
from src.skills.llm_verifier import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    load_transformers_model_call,
    verify_candidates,
)

OUTPUT_DIR = DATA_DIR.parent / "experiments" / "validation" / "extraction"

PILOT_PROG_IDS = [12, 21, 25, 26, 29]
PILOT_JOB_IDS = [1, 20, 193, 253, 429]


def _gather_pilot_rows(dataset: pd.DataFrame) -> list[dict]:
    """Yield row metadata for the 10 pilot docs."""
    progs = dataset[dataset["source_type"] == "programme"].reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    jobs = dataset[dataset["source_type"] == "job_ad"].reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    out: list[dict] = []
    for pid in PILOT_PROG_IDS:
        row = progs.iloc[pid]
        out.append({
            "doc_kind": "programme",
            "doc_id": pid,
            "orig_idx": int(row["_orig_idx"]),
            "title": row["name"],
            "text": row.get("cleaned_text") or row.get("extended_description") or "",
        })
    for jid in PILOT_JOB_IDS:
        row = jobs.iloc[jid]
        out.append({
            "doc_kind": "job_ad",
            "doc_id": jid,
            "orig_idx": int(row["_orig_idx"]),
            "title": row["job_title"],
            "text": row.get("cleaned_text") or row.get("description") or "",
        })
    return out


def main(
    output_dir: Path = OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    suffix: str = "",
    title_aware: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{suffix}" if suffix else ""

    logger.info(f"Loading dataset from {DATASET_PATH}…")
    dataset = pd.read_parquet(DATASET_PATH)
    annotations = pd.concat(
        [
            pd.read_csv(output_dir / "programmes_filled.csv").fillna(""),
            pd.read_csv(output_dir / "jobs_filled.csv").fillna(""),
        ],
        ignore_index=True,
    )

    pilot_rows = _gather_pilot_rows(dataset)
    logger.info(f"Pilot scope: {len(pilot_rows)} documents")

    logger.info(f"Loading verifier model: {model_name}")
    model_call = load_transformers_model_call(model_name=model_name)

    all_decisions: list[dict] = []
    modified_ds = dataset.copy()

    for row in pilot_rows:
        details = _coerce_details(dataset.at[row["orig_idx"], "skill_details"])
        candidates = [
            (d["esco_uri"], d.get("preferred_label", ""))
            for d in details
            if d.get("esco_uri")
        ]
        if not candidates:
            continue

        logger.info(
            f"  Verifying {row['doc_kind']} #{row['doc_id']} ({row['title'][:40]}…) "
            f"— {len(candidates)} URIs in {(len(candidates) + batch_size - 1) // batch_size} batches"
        )
        t0 = time.perf_counter()
        results = verify_candidates(
            doc_text=row["text"],
            candidates=candidates,
            model_call=model_call,
            batch_size=batch_size,
            doc_title=row["title"] if title_aware else None,
            doc_kind=row["doc_kind"] if title_aware else None,
        )
        elapsed = time.perf_counter() - t0
        kept_count = sum(1 for r in results if r.kept)
        logger.info(f"    kept {kept_count}/{len(results)} URIs ({elapsed:.1f}s)")

        kept_uri_set = {r.esco_uri for r in results if r.kept}
        for d in details:
            all_decisions.append({
                "doc_kind": row["doc_kind"],
                "doc_id": row["doc_id"],
                "esco_uri": d.get("esco_uri", ""),
                "preferred_label": d.get("preferred_label", ""),
                "explicit": bool(d.get("explicit", False)),
                "kept": d.get("esco_uri", "") in kept_uri_set,
            })

        new_details = [d for d in details if d.get("esco_uri", "") in kept_uri_set]
        modified_ds.at[row["orig_idx"], "skill_details"] = new_details

    decisions_df = pd.DataFrame(all_decisions)
    decisions_df.to_csv(output_dir / f"llm_verify_decisions{sfx}.csv", index=False)

    cmps, _ = compare_sample(annotations, modified_ds)
    agg = aggregate_metrics(cmps)
    agg.to_csv(output_dir / f"llm_verify_metrics{sfx}.csv", index=False)

    # Report
    lines: list[str] = [f"# LLM-verifier results — model `{model_name}`", ""]
    lines.append("Per-document metrics after applying the local-LLM verifier on the canonical extractor output.")
    lines.append("")
    lines.append("| doc_kind | doc_id | title | gold | kept | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in agg.iterrows():
        title = str(r["doc_title"])[:50]
        lines.append(
            f"| {r['doc_kind']} | {r['doc_id']} | {title} | {r['gold_n']} | {r['extracted_n']} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    (output_dir / f"LLM_VERIFY_REPORT{sfx}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    micro = agg[agg["doc_kind"] == "ALL"].iloc[0]
    logger.info(
        f"Done. Micro-averaged: P={micro['precision']:.3f}, R={micro['recall']:.3f}, "
        f"F1={micro['f1']:.3f}  (TP={int(micro['tp'])} FP={int(micro['fp'])} FN={int(micro['fn'])})"
    )
    logger.info(f"Outputs in {output_dir}/llm_verify_*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--suffix", type=str, default="", help="Appended to output filenames, e.g. 'qwen2.5-3B_v2_title'.")
    parser.add_argument("--title-aware", action="store_true", help="Inject document title + type into the verifier prompt.")
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        suffix=args.suffix,
        title_aware=args.title_aware,
    )
