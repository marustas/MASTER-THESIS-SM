"""
Step 42 — F1 high-IDF blend sweep.

Sweeps mu over a coarse grid, records aggregate metrics per mu, and saves the
per-mu top-1 picks for downstream comparison against the Step 41 baseline.

Outputs:
- experiments/results/evaluation/f1_high_idf_blend/sweep.json
- experiments/results/evaluation/f1_high_idf_blend/top1_per_mu.parquet
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.alignment.hybrid import align_hybrid

DATASET = Path("data/dataset/dataset.parquet")
OUT_DIR = Path("experiments/results/evaluation/f1_high_idf_blend")
SWEEP_JSON = OUT_DIR / "sweep.json"
TOP1_PARQUET = OUT_DIR / "top1_per_mu.parquet"

MU_GRID = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
HEAD_TIE_GAP = 0.02
GENERALIST_FREQ = 5


def metrics_per_mu(rankings: pd.DataFrame, n_prog: int) -> dict:
    by_prog = rankings.sort_values(
        ["programme_id", "hybrid_score"], ascending=[True, False]
    )
    top1 = by_prog.groupby("programme_id").head(1).reset_index(drop=True)
    top2 = by_prog.groupby("programme_id").head(2).reset_index(drop=True)
    top5 = by_prog.groupby("programme_id").head(5).reset_index(drop=True)
    top10 = by_prog.groupby("programme_id").head(10).reset_index(drop=True)

    # Head-tie: top-1 vs top-2 score gap < threshold
    pivot = (
        top2.assign(_rank=top2.groupby("programme_id").cumcount())
        .pivot_table(index="programme_id", columns="_rank", values="hybrid_score")
    )
    head_gaps = (pivot[0] - pivot[1]).fillna(np.inf)
    head_tied = int((head_gaps < HEAD_TIE_GAP).sum())

    # Top-5 generalists — jobs appearing > GENERALIST_FREQ times in top-5
    top5_freq = top5["job_id"].value_counts()
    generalists = int((top5_freq > GENERALIST_FREQ).sum())
    max_top5_freq = int(top5_freq.max()) if len(top5_freq) else 0

    # Top-10 unique
    top10_unique = int(top10["job_id"].nunique())

    return {
        "top1_unique": int(top1["job_id"].nunique()),
        "top1_unique_frac": round(top1["job_id"].nunique() / n_prog, 3),
        "top1_score_mean": round(float(top1["hybrid_score"].mean()), 4),
        "top1_score_median": round(float(top1["hybrid_score"].median()), 4),
        "top1_score_max": round(float(top1["hybrid_score"].max()), 4),
        "head_tied_lt_002": head_tied,
        "top5_generalists_freq_gt5": generalists,
        "top5_max_freq": max_top5_freq,
        "top10_unique": top10_unique,
        "n_pairs": int(len(rankings)),
    }


def jaccard_top10(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Mean per-programme Jaccard over top-10 sets."""
    a_top10 = a.sort_values(
        ["programme_id", "hybrid_score"], ascending=[True, False]
    ).groupby("programme_id").head(10)
    b_top10 = b.sort_values(
        ["programme_id", "hybrid_score"], ascending=[True, False]
    ).groupby("programme_id").head(10)
    a_sets = a_top10.groupby("programme_id")["job_id"].apply(set)
    b_sets = b_top10.groupby("programme_id")["job_id"].apply(set)
    pids = a_sets.index.intersection(b_sets.index)
    js = []
    for p in pids:
        inter = len(a_sets[p] & b_sets[p])
        union = len(a_sets[p] | b_sets[p])
        js.append(inter / union if union else 0.0)
    return round(float(np.mean(js)), 4) if js else 0.0


def main() -> None:
    df = pd.read_parquet(DATASET)
    n_prog = (df["source_type"] == "programme").sum()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict] = []
    top1_rows: list[pd.DataFrame] = []
    baseline = None

    for mu in MU_GRID:
        logger.info(f"=== mu = {mu} ===")
        rankings = align_hybrid(df, hi_idf_f1_lambda=mu)

        m = metrics_per_mu(rankings, n_prog)
        m["mu"] = mu
        if baseline is None:
            baseline = rankings
            m["jaccard_top10_vs_baseline"] = 1.0
        else:
            m["jaccard_top10_vs_baseline"] = jaccard_top10(rankings, baseline)
        sweep_rows.append(m)

        top1 = (
            rankings.sort_values(
                ["programme_id", "hybrid_score"], ascending=[True, False]
            )
            .groupby("programme_id")
            .head(1)
            .copy()
        )
        top1["mu"] = mu
        top1_rows.append(
            top1[["mu", "programme_id", "programme_name", "job_id",
                  "job_title", "hybrid_score"]]
        )

    sweep_df = pd.DataFrame(sweep_rows)
    SWEEP_JSON.write_text(sweep_df.to_json(orient="records", indent=2))
    pd.concat(top1_rows, ignore_index=True).to_parquet(TOP1_PARQUET, index=False)

    print(sweep_df[
        [
            "mu", "top1_unique", "head_tied_lt_002", "top5_generalists_freq_gt5",
            "top5_max_freq", "top1_score_mean", "top1_score_max",
            "jaccard_top10_vs_baseline",
        ]
    ].to_string(index=False))

    print(f"\nSaved sweep → {SWEEP_JSON}")
    print(f"Saved top-1 per mu → {TOP1_PARQUET}")


if __name__ == "__main__":
    main()
