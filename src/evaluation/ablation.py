"""
Step 20 — Extraction Ablation Study.

Removes each of the four explicit-extraction modules (S1 NER, S2 PoS,
S3 Dictionary, S4 Embedding) one at a time, re-extracts skills with the
ablated ensemble, and re-runs symbolic alignment to measure each module's
contribution to alignment quality.

Metrics per configuration:
  - mean weighted Jaccard (all pairs and top-N)
  - mean overlap coefficient
  - mean skills per record (programmes / jobs separately)
  - skill gap coverage (unique gap URIs in top-N)

Output  (experiments/results/ablation/):
  ablation_results.json   — per-config metrics
  ablation_rankings/      — rankings.parquet per config (optional)

Usage:
    python -m src.evaluation.ablation
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from loguru import logger

from src.alignment.symbolic import (
    align_symbolic,
    _build_weighted_skills,
)
from src.scraping.config import DATA_DIR
from src.skills.explicit_extractor import (
    ExplicitSkillExtractor,
    _W_NER,
    _W_POS,
    _W_DICT,
    _W_EMBED,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results" / "ablation"

# ── Ablation configurations ────────────────────────────────────────────────────

ABLATION_CONFIGS: dict[str, dict[str, float]] = {
    "baseline": {"S1": _W_NER, "S2": _W_POS, "S3": _W_DICT, "S4": _W_EMBED},
    "no_S1_ner": {"S1": 0, "S2": _W_POS, "S3": _W_DICT, "S4": _W_EMBED},
    "no_S2_pos": {"S1": _W_NER, "S2": 0, "S3": _W_DICT, "S4": _W_EMBED},
    "no_S3_dict": {"S1": _W_NER, "S2": _W_POS, "S3": 0, "S4": _W_EMBED},
    "no_S4_embed": {"S1": _W_NER, "S2": _W_POS, "S3": _W_DICT, "S4": 0},
}


# ── Core logic ─────────────────────────────────────────────────────────────────


def extract_with_config(
    texts: list[str],
    extractor: ExplicitSkillExtractor,
) -> list[list[dict]]:
    """Run explicit extraction on *texts* and return skill_details dicts."""
    all_details: list[list[dict]] = []
    for text in texts:
        skills = extractor.extract(text)
        all_details.append([asdict(s) for s in skills])
    return all_details


def build_ablated_dataset(
    df: pd.DataFrame,
    skill_details_by_source: dict[str, list[list[dict]]],
) -> pd.DataFrame:
    """Replace skill_details in *df* with ablated extraction results.

    Parameters
    ----------
    df:
        Original dataset (must have ``source_type`` column).
    skill_details_by_source:
        ``{"programme": [...], "job_ad": [...]}`` — one list of skill_details
        per record, in the same order as they appear in *df*.
    """
    df = df.copy()
    prog_mask = df["source_type"] == "programme"
    job_mask = df["source_type"] == "job_ad"

    df.loc[prog_mask, "skill_details"] = pd.Series(
        skill_details_by_source["programme"],
        index=df.loc[prog_mask].index,
    )
    df.loc[job_mask, "skill_details"] = pd.Series(
        skill_details_by_source["job_ad"],
        index=df.loc[job_mask].index,
    )
    return df


def compute_ablation_metrics(
    rankings: pd.DataFrame,
    skill_gaps: pd.DataFrame,
    df_ablated: pd.DataFrame,
    top_n: int,
) -> dict:
    """Compute summary metrics for one ablation configuration."""
    top = rankings.groupby("programme_id").head(top_n)

    prog_mask = df_ablated["source_type"] == "programme"
    job_mask = df_ablated["source_type"] == "job_ad"

    def _skills_count(details) -> int:
        if details is None:
            return 0
        return len(details) if hasattr(details, "__len__") else 0

    prog_skills_counts = df_ablated.loc[prog_mask, "skill_details"].apply(_skills_count)
    job_skills_counts = df_ablated.loc[job_mask, "skill_details"].apply(_skills_count)

    return {
        "weighted_jaccard_mean_all": float(rankings["weighted_jaccard"].mean()),
        "weighted_jaccard_mean_top_n": float(top["weighted_jaccard"].mean()),
        "overlap_coeff_mean_all": float(rankings["overlap_coeff"].mean()),
        "overlap_coeff_mean_top_n": float(top["overlap_coeff"].mean()),
        "mean_skills_programme": float(prog_skills_counts.mean()),
        "mean_skills_job": float(job_skills_counts.mean()),
        "unique_gap_uris": int(skill_gaps["gap_uri"].nunique()) if len(skill_gaps) > 0 else 0,
        "total_gap_entries": int(len(skill_gaps)),
    }


def run_ablation_study(
    df: pd.DataFrame,
    extractor_factory,
    configs: dict[str, dict[str, float]] | None = None,
    top_n: int = 20,
) -> dict[str, dict]:
    """Run the full ablation study.

    Parameters
    ----------
    df:
        Unified dataset with ``source_type``, ``cleaned_text``, ``name``/``job_title``.
    extractor_factory:
        Callable ``(module_weights) -> ExplicitSkillExtractor``.
    configs:
        Ablation configurations.  Defaults to :data:`ABLATION_CONFIGS`.
    top_n:
        Number of top matches for skill gap analysis.

    Returns
    -------
    dict mapping config name → metrics dict.
    """
    if configs is None:
        configs = ABLATION_CONFIGS

    programmes = df[df["source_type"] == "programme"]
    jobs = df[df["source_type"] == "job_ad"]
    prog_texts = programmes["cleaned_text"].tolist()
    job_texts = jobs["cleaned_text"].tolist()

    results: dict[str, dict] = {}

    for config_name, weights in configs.items():
        logger.info(f"Ablation config: {config_name} — weights {weights}")

        extractor = extractor_factory(weights)

        prog_details = extract_with_config(prog_texts, extractor)
        job_details = extract_with_config(job_texts, extractor)

        df_ablated = build_ablated_dataset(
            df,
            {"programme": prog_details, "job_ad": job_details},
        )

        rankings, skill_gaps = align_symbolic(df_ablated, top_n=top_n)
        metrics = compute_ablation_metrics(rankings, skill_gaps, df_ablated, top_n)
        metrics["weights"] = weights
        results[config_name] = metrics

        logger.info(
            f"  {config_name}: jaccard_mean={metrics['weighted_jaccard_mean_all']:.4f}, "
            f"skills_prog={metrics['mean_skills_programme']:.1f}, "
            f"skills_job={metrics['mean_skills_job']:.1f}"
        )

    return results


def compute_deltas(results: dict[str, dict]) -> dict[str, dict]:
    """Compute change from baseline for each ablated config."""
    baseline = results.get("baseline")
    if baseline is None:
        return {}

    metric_keys = [
        "weighted_jaccard_mean_all",
        "weighted_jaccard_mean_top_n",
        "overlap_coeff_mean_all",
        "overlap_coeff_mean_top_n",
        "mean_skills_programme",
        "mean_skills_job",
        "unique_gap_uris",
    ]

    deltas: dict[str, dict] = {}
    for name, metrics in results.items():
        if name == "baseline":
            continue
        delta = {}
        for key in metric_keys:
            b_val = baseline[key]
            a_val = metrics[key]
            delta[key] = round(a_val - b_val, 6)
            if b_val != 0:
                delta[f"{key}_pct"] = round((a_val - b_val) / abs(b_val) * 100, 2)
        deltas[name] = delta

    return deltas


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR,
    top_n: int = 20,
) -> None:
    """Load dataset, run ablation study, persist results."""
    from src.skills.esco_loader import load_esco_index

    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_parquet(dataset_path)

    esco_index = load_esco_index()

    def extractor_factory(weights: dict[str, float]) -> ExplicitSkillExtractor:
        return ExplicitSkillExtractor(
            esco_index=esco_index,
            module_weights=weights,
        )

    results = run_ablation_study(df, extractor_factory, top_n=top_n)
    deltas = compute_deltas(results)

    output_dir.mkdir(parents=True, exist_ok=True)

    output = {"configs": results, "deltas": deltas}
    out_path = output_dir / "ablation_results.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    logger.info(f"Ablation results → {out_path}")


if __name__ == "__main__":
    run()
