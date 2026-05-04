"""
Step 37 — Confidence-weighted implicit skill weighting experiment.

Compares three implicit-skill weighting modes in the hybrid alignment:

  uniform : every implicit skill gets the paper's flat E3 = 0.5 (baseline).
  linear  : 0.5 × clip((conf - 0.70) / 0.30, 0, 1).
  sqrt    : 0.5 × sqrt(clip(...)).  Gentler decay near the floor.

Reports per-mode:
  * top-1 unique count and diversity ratio
  * top-1 score mean / max / std / CoV
  * top-1 vs top-2 gap distribution (head-discrimination metric)
  * cross-strategy Spearman vs symbolic and semantic top-50
  * generalist concentration (max top-5 repeat)

Output:
  experiments/results/exp_implicit_confidence/
    summary.json                 — all metrics, all modes, deltas vs baseline
    rankings_<mode>.parquet      — full ranking per mode
    FINDINGS.md                  — narrative comparison table

Usage:
  .venv/bin/python -m src.evaluation.implicit_confidence_experiment
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from src.alignment.hybrid import align_hybrid
from src.alignment.semantic import align_semantic
from src.alignment.symbolic import align_symbolic_weighted
from src.scraping.config import DATA_DIR

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results" / "exp_implicit_confidence"


MODES = ("uniform", "linear", "sqrt")


# ── Metrics ────────────────────────────────────────────────────────────────────


def _compute_metrics(
    rankings: pd.DataFrame,
    sym_rankings: pd.DataFrame,
    sem_rankings: pd.DataFrame,
) -> dict[str, Any]:
    """Compute headline metrics for a single hybrid ranking."""
    # ── Top-1 ────────────────────────────────────────────────────────────────
    top1 = (
        rankings.sort_values(["programme_id", "hybrid_score"], ascending=[True, False])
        .groupby("programme_id")
        .head(1)
    )
    n_prog = top1["programme_id"].nunique()
    unique_top1 = top1["job_id"].nunique()
    top1_counts = top1["job_id"].value_counts()
    top1_max_repeat = int(top1_counts.max()) if len(top1_counts) else 0

    # ── Top-1 vs Top-2 gap (head discrimination) ─────────────────────────────
    head = (
        rankings.sort_values(["programme_id", "hybrid_score"], ascending=[True, False])
        .groupby("programme_id")
        .head(2)
    )
    gaps = head.groupby("programme_id")["hybrid_score"].agg(
        lambda s: float(s.iloc[0] - s.iloc[1]) if len(s) >= 2 else 0.0
    )
    gap_lt_002 = int((gaps < 0.02).sum())
    gap_lt_005 = int((gaps < 0.05).sum())

    # ── Top-5 generalists ────────────────────────────────────────────────────
    top5 = (
        rankings.sort_values(["programme_id", "hybrid_score"], ascending=[True, False])
        .groupby("programme_id")
        .head(5)
    )
    top5_counts = top5["job_id"].value_counts()
    top5_max_repeat = int(top5_counts.max()) if len(top5_counts) else 0
    top5_generalists = int((top5_counts > 5).sum())

    # ── Cross-strategy Spearman (per-programme, mean) ────────────────────────
    sym_lookup = sym_rankings.set_index(["programme_id", "job_id"])["programme_recall"]
    sem_lookup = sem_rankings.set_index(["programme_id", "job_id"])["cosine_combined"]
    spearman_sym = []
    spearman_sem = []
    for pid, grp in rankings.groupby("programme_id"):
        if len(grp) < 5:
            continue
        idx = list(zip(grp["programme_id"], grp["job_id"]))
        try:
            sym_scores = sym_lookup.reindex(idx).fillna(0.0).values
            sem_scores = sem_lookup.reindex(idx).fillna(0.0).values
            hyb_scores = grp["hybrid_score"].values
            if np.std(sym_scores) > 0 and np.std(hyb_scores) > 0:
                spearman_sym.append(spearmanr(sym_scores, hyb_scores).correlation)
            if np.std(sem_scores) > 0 and np.std(hyb_scores) > 0:
                spearman_sem.append(spearmanr(sem_scores, hyb_scores).correlation)
        except Exception:
            continue

    return {
        "n_programmes": int(n_prog),
        # diversity
        "unique_top1": int(unique_top1),
        "top1_diversity": round(unique_top1 / max(n_prog, 1), 4),
        "top1_max_repeat": top1_max_repeat,
        "top5_generalists": top5_generalists,
        "top5_max_repeat": top5_max_repeat,
        # score distribution at the head
        "top1_score_mean": float(top1["hybrid_score"].mean()),
        "top1_score_max": float(top1["hybrid_score"].max()),
        "top1_score_min": float(top1["hybrid_score"].min()),
        "top1_score_std": float(top1["hybrid_score"].std()),
        "top1_score_cov": float(top1["hybrid_score"].std() / max(top1["hybrid_score"].mean(), 1e-9)),
        # head discrimination (top-1 vs top-2)
        "gap_mean": float(gaps.mean()),
        "gap_median": float(gaps.median()),
        "gap_p10": float(np.percentile(gaps, 10)),
        "gap_lt_002": gap_lt_002,
        "gap_lt_005": gap_lt_005,
        # cross-strategy
        "spearman_sym_hyb_mean": float(np.mean(spearman_sym)) if spearman_sym else None,
        "spearman_sem_hyb_mean": float(np.mean(spearman_sem)) if spearman_sem else None,
    }


# ── Experiment ────────────────────────────────────────────────────────────────


def run_experiment(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Run hybrid alignment under all three implicit-confidence modes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)

    # Semantic ranking is mode-independent (uses embeddings only)
    logger.info("Computing semantic rankings (shared across modes)…")
    sem_full = align_semantic(df)[["programme_id", "job_id", "cosine_combined"]]

    results: dict[str, Any] = {"by_mode": {}, "deltas_vs_uniform": {}}

    for mode in MODES:
        logger.info(f"━━━ mode = {mode} ━━━")

        # Symbolic (mode-dependent) — for Spearman vs hybrid
        sym_rankings, _ = align_symbolic_weighted(
            df,
            top_n=50,
            use_programme_idf=True,
            implicit_confidence_mode=mode,
        )
        sym_rankings = sym_rankings[["programme_id", "job_id", "programme_recall"]]

        # Hybrid (mode-dependent)
        hybrid = align_hybrid(
            df,
            semantic_top_n=50,
            alpha=0.55,
            ipf_top_k=30,
            ipf_floor=0.1,
            ipf_strict_floor=0.05,
            ipf_strict_threshold=0.5,
            norm_confidence=True,
            gamma=0.3,
            use_programme_idf=True,
            implicit_confidence_mode=mode,
        )
        hybrid.to_parquet(output_dir / f"rankings_{mode}.parquet", index=False)

        metrics = _compute_metrics(hybrid, sym_rankings, sem_full)
        results["by_mode"][mode] = metrics

    # Deltas vs uniform baseline
    base = results["by_mode"]["uniform"]
    for mode in ("linear", "sqrt"):
        m = results["by_mode"][mode]
        results["deltas_vs_uniform"][mode] = {
            "unique_top1": m["unique_top1"] - base["unique_top1"],
            "top1_score_mean": m["top1_score_mean"] - base["top1_score_mean"],
            "top1_score_max": m["top1_score_max"] - base["top1_score_max"],
            "gap_mean": m["gap_mean"] - base["gap_mean"],
            "gap_lt_002": m["gap_lt_002"] - base["gap_lt_002"],
            "top1_score_cov": m["top1_score_cov"] - base["top1_score_cov"],
            "spearman_sym_hyb_mean": (
                (m["spearman_sym_hyb_mean"] or 0.0)
                - (base["spearman_sym_hyb_mean"] or 0.0)
            ),
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Summary → {summary_path}")

    return results


def _format_table(results: dict[str, Any]) -> str:
    """Pretty-print a markdown comparison table for the FINDINGS file."""
    rows = [
        ("Top-1 unique",          "unique_top1"),
        ("Top-1 diversity",       "top1_diversity"),
        ("Top-1 max repeat",      "top1_max_repeat"),
        ("Top-5 generalists (>5)","top5_generalists"),
        ("Top-1 score mean",      "top1_score_mean"),
        ("Top-1 score max",       "top1_score_max"),
        ("Top-1 score CoV",       "top1_score_cov"),
        ("Gap top1↔top2 mean",   "gap_mean"),
        ("Programmes with gap<0.02","gap_lt_002"),
        ("Programmes with gap<0.05","gap_lt_005"),
        ("Spearman sym↔hyb",     "spearman_sym_hyb_mean"),
        ("Spearman sem↔hyb",     "spearman_sem_hyb_mean"),
    ]

    lines = ["| Metric | uniform (baseline) | linear | sqrt |", "|---|---|---|---|"]
    for label, key in rows:
        vals = []
        for m in MODES:
            v = results["by_mode"][m].get(key)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")
    return "\n".join(lines)


def write_findings(results: dict[str, Any], output_dir: Path = RESULTS_DIR) -> None:
    """Write a markdown FINDINGS.md summarising the experiment."""
    table = _format_table(results)
    body = f"""# Step 37 — Confidence-weighted Implicit Skills

## Setup

Three implicit-skill weighting modes compared in the hybrid alignment
(α = 0.55, γ = 0.3, semantic_top_n = 50, IPF on, programme IDF on):

| Mode    | Implicit weight formula |
|---------|--------------------------|
| uniform | `0.5` (paper baseline)  |
| linear  | `0.5 × clip((conf − 0.70) / 0.30, 0, 1)` |
| sqrt    | `0.5 × sqrt(clip((conf − 0.70) / 0.30, 0, 1))` |

Implicit confidence = propagation cosine to source neighbour, range
0.70–1.00 after Step 4b filtering (mean ≈ 0.77, p50 ≈ 0.74).

## Results

{table}

## Deltas vs uniform baseline

| Mode | Δ unique_top1 | Δ top1_score_mean | Δ gap_mean | Δ gap<0.02 | Δ Spearman sym↔hyb |
|---|---|---|---|---|---|
| linear | {results['deltas_vs_uniform']['linear']['unique_top1']:+d} | {results['deltas_vs_uniform']['linear']['top1_score_mean']:+.4f} | {results['deltas_vs_uniform']['linear']['gap_mean']:+.4f} | {results['deltas_vs_uniform']['linear']['gap_lt_002']:+d} | {results['deltas_vs_uniform']['linear']['spearman_sym_hyb_mean']:+.4f} |
| sqrt   | {results['deltas_vs_uniform']['sqrt']['unique_top1']:+d} | {results['deltas_vs_uniform']['sqrt']['top1_score_mean']:+.4f} | {results['deltas_vs_uniform']['sqrt']['gap_mean']:+.4f} | {results['deltas_vs_uniform']['sqrt']['gap_lt_002']:+d} | {results['deltas_vs_uniform']['sqrt']['spearman_sym_hyb_mean']:+.4f} |

## Notes

* Diagnostic: implicit confidence is heavily concentrated near the floor
  (p50 ≈ 0.74 → linear factor 0.067, sqrt factor 0.18). Linear therefore
  effectively suppresses the implicit channel by ≈ 8× at the median
  confidence; sqrt by ≈ 3×.

* The decision is whether the cleaner methodological story
  (confidence-aware weighting) justifies the loss in implicit signal mass.
  The numbers above answer this empirically: head discrimination is the
  primary metric to watch, with diversity as the negative guardrail.
"""
    (output_dir / "FINDINGS.md").write_text(body)
    logger.info(f"FINDINGS → {output_dir / 'FINDINGS.md'}")


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    out = run_experiment()
    write_findings(out)
    # Echo a compact comparison to stdout
    print("\n" + _format_table(out) + "\n")
