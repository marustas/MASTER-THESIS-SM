"""
End-to-end pipeline orchestrator.

Runs all steps from data collection to curriculum recommendations.
Skips any step whose sentinel output already exists, unless --force is set.

Steps:
  1   scrape_programmes   data/raw/programmes/lama_bpo_programmes.json
  2   scrape_jobs         data/raw/job_ads/all_jobs.json
  3   preprocess          data/processed/programmes/programmes_preprocessed.parquet
  4   extract_skills      data/processed/programmes/programmes_with_skills.parquet
  5   embed               data/embeddings/programmes_embeddings.parquet
  6   build_dataset       data/dataset/dataset.parquet
  7   cluster             data/dataset/dataset.parquet  (cluster_label column)
  8   align_symbolic      experiments/results/exp1_symbolic/rankings.parquet
  9   align_semantic      experiments/results/exp2_semantic/rankings.parquet
  10  align_hybrid        experiments/results/exp3_hybrid/rankings.parquet
  11  evaluate            experiments/results/evaluation/per_programme.parquet
  12  recommend           experiments/results/recommendations/programme_recommendations.parquet

Usage:
    python -m src.pipeline                    # run all steps
    python -m src.pipeline --from 8           # start from step 8
    python -m src.pipeline --steps 3,4,5      # run specific steps only
    python -m src.pipeline --force            # ignore existing outputs
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

from src.scraping.config import DATA_DIR

logger = logging.getLogger(__name__)

RESULTS_DIR   = Path("experiments/results")
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DATASET_PATH  = DATA_DIR / "dataset" / "dataset.parquet"

# ── Sentinel files — presence signals a completed step ────────────────────────
# Step 7 is handled separately (checks for cluster_label column in dataset).
_SENTINEL: dict[int, Path] = {
    1:  DATA_DIR / "raw" / "programmes" / "lama_bpo_programmes.json",
    2:  DATA_DIR / "raw" / "job_ads"    / "all_jobs.json",
    3:  PROCESSED_DIR / "programmes" / "programmes_preprocessed.parquet",
    4:  PROCESSED_DIR / "programmes" / "programmes_with_skills.parquet",
    5:  EMBEDDINGS_DIR / "programmes_embeddings.parquet",
    6:  DATASET_PATH,
    8:  RESULTS_DIR / "exp1_symbolic"   / "rankings.parquet",
    9:  RESULTS_DIR / "exp2_semantic"   / "rankings.parquet",
    10: RESULTS_DIR / "exp3_hybrid"     / "rankings.parquet",
    11: RESULTS_DIR / "evaluation"      / "per_programme.parquet",
    12: RESULTS_DIR / "recommendations" / "programme_recommendations.parquet",
}

ALL_STEPS = list(range(1, 13))


# ── Step completion checks ─────────────────────────────────────────────────────

def _step_done(step: int) -> bool:
    if step == 7:
        if not DATASET_PATH.exists():
            return False
        schema = pq.read_schema(DATASET_PATH)
        return "cluster_label" in schema.names
    return _SENTINEL[step].exists()


# ── Step runners ───────────────────────────────────────────────────────────────

def _run_step(step: int) -> None:
    if step == 1:
        from src.scraping.lama_bpo import run as _run
        asyncio.run(_run())

    elif step == 2:
        from src.scraping.job_ads import run as _run
        asyncio.run(_run())

    elif step == 3:
        from src.preprocessing.pipeline import run as _run
        _run()

    elif step == 4:
        from src.skills.skill_mapper import run as _run
        _run()
        from src.skills.skill_filter import run as _filter
        _filter()

    elif step == 5:
        from src.embeddings.generator import run as _run
        _run()

    elif step == 6:
        from src.dataset_builder import build as _run
        _run()

    elif step == 7:
        from src.clustering.programme_clustering import run as cluster_programmes
        from src.clustering.job_clustering import run as cluster_jobs
        cluster_programmes()
        cluster_jobs()

    elif step == 8:
        from src.alignment.symbolic import run_symbolic_alignment as _run
        _run()

    elif step == 9:
        from src.alignment.semantic import run_semantic_alignment as _run
        _run()

    elif step == 10:
        from src.alignment.hybrid import run_hybrid_alignment as _run
        _run()

    elif step == 11:
        from src.evaluation.cross_strategy import run_evaluation as _run
        _run()

    elif step == 12:
        from src.recommendations.generator import run_recommendations as _run
        _run()

    else:
        raise ValueError(f"Unknown step: {step}")


# ── Orchestration ──────────────────────────────────────────────────────────────

def _preload_hf_models(steps: list[int]) -> None:
    """
    Download / warm-up HuggingFace models before any asyncio.run() call,
    then switch HuggingFace Hub to offline mode.

    huggingface_hub uses httpx internally.  After asyncio.run() closes its
    event loop the httpx client is left in a broken state, causing
    "Cannot send a request, as the client has been closed" on every
    subsequent model-check or download attempt.  Pre-loading here ensures
    models are cached; setting HF_HUB_OFFLINE=1 afterwards prevents any
    further network calls so later steps load from disk only.
    """
    import os

    needs_translation = any(s in steps for s in range(3, 13))
    needs_sentence_transformer = any(s in steps for s in range(4, 13))

    if needs_translation:
        logger.info("Pre-loading translation model…")
        from src.preprocessing.translate import translate_lt_to_en
        translate_lt_to_en("testas")

    if needs_sentence_transformer:
        logger.info("Pre-loading sentence-transformer model…")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2")

    if needs_translation or needs_sentence_transformer:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info("HuggingFace Hub set to offline mode (models cached)")


def run_pipeline(steps: list[int], force: bool = False) -> None:
    """Run the given steps in order, skipping completed ones unless force=True."""
    _preload_hf_models(steps)
    for step in steps:
        if not force and _step_done(step):
            logger.info(f"Step {step:2d} — skipped (output exists)")
            continue
        logger.info(f"Step {step:2d} — running…")
        _run_step(step)
        logger.info(f"Step {step:2d} — done")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full study-programme alignment pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from",
        dest="from_step",
        type=int,
        metavar="N",
        help="Start from step N (inclusive), run all following steps.",
    )
    group.add_argument(
        "--steps",
        type=str,
        metavar="N,N,...",
        help="Run only the specified steps, comma-separated (e.g. 3,4,5).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even when their output already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)

    if args.steps:
        try:
            steps = [int(s.strip()) for s in args.steps.split(",")]
        except ValueError:
            logger.error("--steps requires a comma-separated list of integers, e.g. 3,4,5")
            sys.exit(1)
        invalid = [s for s in steps if s not in ALL_STEPS]
        if invalid:
            logger.error(f"Unknown step(s): {invalid}. Valid range: 1–12.")
            sys.exit(1)
    elif args.from_step is not None:
        if args.from_step not in ALL_STEPS:
            logger.error(f"--from must be between 1 and 12, got {args.from_step}.")
            sys.exit(1)
        steps = [s for s in ALL_STEPS if s >= args.from_step]
    else:
        steps = ALL_STEPS

    run_pipeline(steps, force=args.force)


if __name__ == "__main__":
    main()
