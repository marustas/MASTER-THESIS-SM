"""
Step 38 — Cross-encoder re-ranking experiment.

Compares three hybrid configurations on the full dataset:

  baseline       : current hybrid (α = 0.55, no cross-encoder).
  replace_cos    : drop the bi-encoder cosine channel entirely
                   (α = 0, xe_alpha = 0.55, recall_weight = 0.45).
  three_channel  : split the old cosine half between cos and xe
                   (α = 0.275, xe_alpha = 0.275, recall_weight = 0.45).

Cross-encoder: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (default).

Reports per-config:
  * top-1 unique count and diversity ratio
  * top-1 score mean / max / std / CoV
  * top-1 vs top-2 gap distribution (head-discrimination metric)
  * generalist concentration (max top-5 repeat)
  * Spearman vs symbolic, semantic, and the baseline hybrid

Output:
  experiments/results/exp3_hybrid_xenc/
    summary.json                 — all metrics per config + deltas vs baseline
    rankings_<config>.parquet    — full ranking per config
    FINDINGS.md                  — narrative comparison

Usage:
  .venv/bin/python -m src.evaluation.cross_encoder_experiment
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from src.alignment.cross_encoder import DEFAULT_MODEL, load_cross_encoder
from src.alignment.hybrid import align_hybrid
from src.alignment.semantic import align_semantic
from src.alignment.symbolic import align_symbolic_weighted
from src.scraping.config import DATA_DIR

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results" / "exp3_hybrid_xenc"


CONFIGS: dict[str, dict] = {
    "baseline":               {"alpha": 0.55,  "xe_alpha": 0.0,   "xe_pool_mode": "single"},
    "three_channel":          {"alpha": 0.275, "xe_alpha": 0.275, "xe_pool_mode": "single"},
    "three_channel_secwm":    {"alpha": 0.275, "xe_alpha": 0.275, "xe_pool_mode": "section_weighted"},
    "three_channel_secmax":   {"alpha": 0.275, "xe_alpha": 0.275, "xe_pool_mode": "section_max"},
    "three_channel_jobchunk": {"alpha": 0.275, "xe_alpha": 0.275, "xe_pool_mode": "job_chunked_max"},
    "three_channel_secxjob":  {"alpha": 0.275, "xe_alpha": 0.275, "xe_pool_mode": "section_x_job_max"},
}


# ── Metrics ────────────────────────────────────────────────────────────────────

def _compute_metrics(
    rankings: pd.DataFrame,
    sym_rankings: pd.DataFrame,
    sem_rankings: pd.DataFrame,
    baseline_rankings: pd.DataFrame | None,
) -> dict[str, Any]:
    """Compute headline metrics for a single hybrid ranking."""
    sorted_full = rankings.sort_values(
        ["programme_id", "hybrid_score"], ascending=[True, False]
    )
    top1 = sorted_full.groupby("programme_id").head(1)
    n_prog = top1["programme_id"].nunique()
    unique_top1 = top1["job_id"].nunique()
    top1_counts = top1["job_id"].value_counts()
    top1_max_repeat = int(top1_counts.max()) if len(top1_counts) else 0

    head = sorted_full.groupby("programme_id").head(2)
    gaps = head.groupby("programme_id")["hybrid_score"].agg(
        lambda s: float(s.iloc[0] - s.iloc[1]) if len(s) >= 2 else 0.0
    )
    gap_lt_002 = int((gaps < 0.02).sum())
    gap_lt_005 = int((gaps < 0.05).sum())

    top5 = sorted_full.groupby("programme_id").head(5)
    top5_counts = top5["job_id"].value_counts()
    top5_max_repeat = int(top5_counts.max()) if len(top5_counts) else 0
    top5_generalists = int((top5_counts > 5).sum())

    sym_lookup = sym_rankings.set_index(["programme_id", "job_id"])["programme_recall"]
    sem_lookup = sem_rankings.set_index(["programme_id", "job_id"])["cosine_combined"]
    base_lookup = (
        baseline_rankings.set_index(["programme_id", "job_id"])["hybrid_score"]
        if baseline_rankings is not None else None
    )

    spearman_sym, spearman_sem, spearman_base = [], [], []
    for pid, grp in rankings.groupby("programme_id"):
        if len(grp) < 5:
            continue
        idx = list(zip(grp["programme_id"], grp["job_id"]))
        hyb_scores = grp["hybrid_score"].values
        if np.std(hyb_scores) == 0:
            continue
        try:
            sym_scores = sym_lookup.reindex(idx).fillna(0.0).values
            if np.std(sym_scores) > 0:
                spearman_sym.append(spearmanr(sym_scores, hyb_scores).correlation)
            sem_scores = sem_lookup.reindex(idx).fillna(0.0).values
            if np.std(sem_scores) > 0:
                spearman_sem.append(spearmanr(sem_scores, hyb_scores).correlation)
            if base_lookup is not None:
                base_scores = base_lookup.reindex(idx).fillna(0.0).values
                if np.std(base_scores) > 0:
                    spearman_base.append(spearmanr(base_scores, hyb_scores).correlation)
        except Exception:
            continue

    # Top-1 agreement with baseline
    if baseline_rankings is not None:
        base_top1 = (
            baseline_rankings.sort_values(
                ["programme_id", "hybrid_score"], ascending=[True, False]
            )
            .groupby("programme_id")
            .head(1)[["programme_id", "job_id"]]
            .set_index("programme_id")["job_id"]
        )
        top1_idx = top1.set_index("programme_id")["job_id"]
        common_pids = top1_idx.index.intersection(base_top1.index)
        top1_agreement = int((top1_idx.loc[common_pids] == base_top1.loc[common_pids]).sum())
    else:
        top1_agreement = None

    return {
        "n_programmes": int(n_prog),
        "unique_top1": int(unique_top1),
        "top1_diversity": round(unique_top1 / max(n_prog, 1), 4),
        "top1_max_repeat": top1_max_repeat,
        "top5_generalists": top5_generalists,
        "top5_max_repeat": top5_max_repeat,
        "top1_score_mean": float(top1["hybrid_score"].mean()),
        "top1_score_max": float(top1["hybrid_score"].max()),
        "top1_score_min": float(top1["hybrid_score"].min()),
        "top1_score_std": float(top1["hybrid_score"].std()),
        "top1_score_cov": float(
            top1["hybrid_score"].std() / max(top1["hybrid_score"].mean(), 1e-9)
        ),
        "gap_mean": float(gaps.mean()),
        "gap_median": float(gaps.median()),
        "gap_p10": float(np.percentile(gaps, 10)),
        "gap_lt_002": gap_lt_002,
        "gap_lt_005": gap_lt_005,
        "spearman_sym_hyb_mean": float(np.mean(spearman_sym)) if spearman_sym else None,
        "spearman_sem_hyb_mean": float(np.mean(spearman_sem)) if spearman_sem else None,
        "spearman_base_hyb_mean":
            float(np.mean(spearman_base)) if spearman_base else None,
        "top1_agreement_with_baseline": top1_agreement,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR,
    cross_encoder_model: str | object = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Run hybrid alignment under all three cross-encoder configs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)

    logger.info("Computing semantic + symbolic rankings (shared across configs)…")
    sem_full = align_semantic(df)[["programme_id", "job_id", "cosine_combined"]]
    sym_rankings, _ = align_symbolic_weighted(
        df, top_n=50, use_programme_idf=True, implicit_confidence_mode="sqrt",
    )
    sym_rankings = sym_rankings[["programme_id", "job_id", "programme_recall"]]

    # Load the cross-encoder once and reuse — avoids reloading per config
    logger.info(f"Loading cross-encoder once for reuse: {cross_encoder_model}")
    xe_model = load_cross_encoder(cross_encoder_model)

    results: dict[str, Any] = {
        "model": cross_encoder_model if isinstance(cross_encoder_model, str)
                 else type(cross_encoder_model).__name__,
        "by_config": {},
        "deltas_vs_baseline": {},
    }

    rankings_by_config: dict[str, pd.DataFrame] = {}
    for name, cfg in CONFIGS.items():
        logger.info(
            f"━━━ config = {name} "
            f"(alpha={cfg['alpha']}, xe_alpha={cfg['xe_alpha']}, "
            f"pool={cfg['xe_pool_mode']}) ━━━"
        )
        ce_arg = xe_model if cfg["xe_alpha"] > 0 else None
        rankings = align_hybrid(
            df,
            semantic_top_n=50,
            alpha=cfg["alpha"],
            xe_alpha=cfg["xe_alpha"],
            xe_pool_mode=cfg["xe_pool_mode"],
            cross_encoder_model=ce_arg,
            ipf_top_k=30,
            ipf_floor=0.1,
            ipf_strict_floor=0.05,
            ipf_strict_threshold=0.5,
            norm_confidence=True,
            gamma=0.3,
            use_programme_idf=True,
            implicit_confidence_mode="sqrt",
        )
        rankings.to_parquet(output_dir / f"rankings_{name}.parquet", index=False)
        rankings_by_config[name] = rankings

    base = rankings_by_config["baseline"]
    for name, rankings in rankings_by_config.items():
        baseline_for_metrics = None if name == "baseline" else base
        results["by_config"][name] = _compute_metrics(
            rankings, sym_rankings, sem_full, baseline_for_metrics,
        )

    base_metrics = results["by_config"]["baseline"]
    for name in CONFIGS:
        if name == "baseline":
            continue
        m = results["by_config"][name]
        results["deltas_vs_baseline"][name] = {
            "unique_top1": m["unique_top1"] - base_metrics["unique_top1"],
            "top1_score_mean": m["top1_score_mean"] - base_metrics["top1_score_mean"],
            "top1_score_max": m["top1_score_max"] - base_metrics["top1_score_max"],
            "gap_mean": m["gap_mean"] - base_metrics["gap_mean"],
            "gap_lt_002": m["gap_lt_002"] - base_metrics["gap_lt_002"],
            "top1_score_cov": m["top1_score_cov"] - base_metrics["top1_score_cov"],
            "top5_generalists":
                m["top5_generalists"] - base_metrics["top5_generalists"],
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Summary → {summary_path}")

    return results


def _format_table(results: dict[str, Any]) -> str:
    rows = [
        ("Top-1 unique",            "unique_top1"),
        ("Top-1 diversity",         "top1_diversity"),
        ("Top-1 max repeat",        "top1_max_repeat"),
        ("Top-5 generalists (>5)",  "top5_generalists"),
        ("Top-1 score mean",        "top1_score_mean"),
        ("Top-1 score max",         "top1_score_max"),
        ("Top-1 score CoV",         "top1_score_cov"),
        ("Gap top1↔top2 mean",     "gap_mean"),
        ("Programmes with gap<0.02","gap_lt_002"),
        ("Programmes with gap<0.05","gap_lt_005"),
        ("Spearman sym↔hyb",       "spearman_sym_hyb_mean"),
        ("Spearman sem↔hyb",       "spearman_sem_hyb_mean"),
        ("Spearman base↔hyb",      "spearman_base_hyb_mean"),
        ("Top-1 agreement w/ base", "top1_agreement_with_baseline"),
    ]

    headers = list(CONFIGS.keys())
    lines = [
        "| Metric | " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * (len(headers) + 1)) + "|",
    ]
    for label, key in rows:
        vals = []
        for cfg in headers:
            v = results["by_config"][cfg].get(key)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_findings(results: dict[str, Any], output_dir: Path = RESULTS_DIR) -> None:
    table = _format_table(results)
    deltas = results["deltas_vs_baseline"]

    delta_rows = []
    for name, d in deltas.items():
        delta_rows.append(
            f"| {name} | "
            f"{d['unique_top1']:+d} | "
            f"{d['top1_score_mean']:+.4f} | "
            f"{d['gap_mean']:+.4f} | "
            f"{d['gap_lt_002']:+d} | "
            f"{d['top5_generalists']:+d} |"
        )
    delta_table = "\n".join(delta_rows)

    body = f"""# Step 38 — Cross-encoder Re-ranking

## Setup

Hybrid configurations compared on the full dataset
(γ = 0.3, semantic_top_n = 50, IPF on, programme IDF on,
implicit_confidence_mode = "sqrt").

| Config                  | α (cos) | xe_alpha | recall | xe_pool_mode      |
|-------------------------|---------|----------|--------|--------------------|
| baseline                | 0.55    | 0.00     | 0.45   | —                  |
| three_channel           | 0.275   | 0.275    | 0.45   | single             |
| three_channel_secwm     | 0.275   | 0.275    | 0.45   | section_weighted   |
| three_channel_secmax    | 0.275   | 0.275    | 0.45   | section_max        |

Section-aware variants split each programme into the same section groups
used by Step 34's section-weighted embeddings (subjects 0.35, outcomes 0.25,
identity 0.15, specialisations 0.20, _remainder 0.05) and score each
non-empty section against the full job text.  ``section_weighted`` pools
by the section weights; ``section_max`` keeps the strongest-matching
section.  This bypasses the 512-token shared budget of the cross-encoder.

Cross-encoder: `{results['model']}`

## Results

{table}

## Deltas vs baseline

| Config | Δ unique_top1 | Δ top1_score_mean | Δ gap_mean | Δ gap<0.02 | Δ top5_generalists |
|---|---|---|---|---|---|
{delta_table}

## Notes

* Hypothesis under test: bi-encoder cosine has spent its variance during
  Stage 1 retrieval (cosine top-1 == hybrid top-1 in 0/45 programmes
  pre-Step 38). A cross-encoder re-evaluates each pair from scratch with
  full token-level attention, producing fresh ranking signal in the
  candidate pool.

* Single-pass cross-encoder shares its 512-token budget with the job text,
  giving each side ~256 tokens. The section-aware variants lift this
  ceiling by re-scoring per-section.

* Watch metrics in this order:
  1. **Programmes with gap < 0.02** — primary head-discrimination metric.
     Step 38 is justified iff this drops appreciably.
  2. **Top-1 diversity** — guardrail. Must not regress.
  3. **Top-5 generalists** — guardrail. Must not regress.
  4. **Top-1 score mean / max** — informational; cross-encoder scores live
     on a different scale, so absolute mean shifts are expected.

* Spearman base↔hyb measures how much the new config rearranges the old
  ranking. Low Spearman + better head discrimination = the cross-encoder
  is contributing genuinely new signal, not just redecorating the order.
"""
    # Auto-generated summary table — kept as a quick-look reference.
    # FINDINGS.md is curated by hand and must not be overwritten by re-runs.
    (output_dir / "summary_table.md").write_text(body)
    logger.info(f"Summary table → {output_dir / 'summary_table.md'}")


if __name__ == "__main__":
    out = run_experiment()
    write_findings(out)
    print("\n" + _format_table(out) + "\n")
