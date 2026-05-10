"""Histogram of skills-per-programme across the 45 programmes."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATASET = Path("data/dataset/dataset.parquet")
OUT = Path("experiments/results/evaluation/skills_per_programme.png")


def main() -> None:
    df = pd.read_parquet(DATASET)
    prog = df[df["source_type"] == "programme"]
    counts = prog["all_skills"].apply(len).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=range(0, counts.max() + 6, 5), edgecolor="black", color="steelblue")
    ax.axvline(counts.mean(), color="red", linestyle="--", label=f"mean={counts.mean():.1f}")
    ax.axvline(float(pd.Series(counts).median()), color="orange", linestyle="--", label=f"median={pd.Series(counts).median():.0f}")
    ax.set_xlabel("Skills per programme (explicit + implicit)")
    ax.set_ylabel("Number of programmes")
    ax.set_title(f"Skill count distribution — {len(counts)} programmes")
    ax.legend()
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Saved: {OUT}")
    print(f"n={len(counts)}, min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}, median={pd.Series(counts).median():.0f}")


if __name__ == "__main__":
    main()
