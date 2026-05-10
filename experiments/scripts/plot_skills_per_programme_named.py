"""Stacked horizontal bar chart — explicit + implicit skills per named programme."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATASET = Path("data/dataset/dataset.parquet")
OUT = Path("experiments/results/evaluation/skills_per_programme_named.png")

INST_SHORT = {
    "Vilnius University": "VU",
    "Vilnius Gediminas Technical University": "VILNIUS TECH",
    "Kaunas University of Technology": "KTU",
    "Vytautas Magnus University": "VDU",
    "Mykolas Romeris University": "MRU",
    "Klaipėda University": "KU",
    "Vilnius Business College": "VVK",
    "Lithuania Business College": "LBC",
    "SMK College of Applied Sciences": "SMK",
    "Utena College": "Utena",
}


def short_inst(inst: str) -> str:
    if inst in INST_SHORT:
        return INST_SHORT[inst]
    inst = re.sub(r"\s*/\s*Higher Education Institution.*", "", inst)
    inst = re.sub(r"/State Higher Education Institution", "", inst)
    return inst


def main() -> None:
    df = pd.read_parquet(DATASET)
    prog = df[df["source_type"] == "programme"].copy()
    prog["n_explicit"] = prog["explicit_skills"].apply(len)
    prog["n_implicit"] = prog["implicit_skills"].apply(len)
    prog["n_total"] = prog["all_skills"].apply(len)
    prog["label"] = prog.apply(lambda r: f"{r['name']}  ({short_inst(r['institution'])})", axis=1)
    prog = prog.sort_values("n_total", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 14))
    y = range(len(prog))
    ax.barh(y, prog["n_explicit"], color="steelblue", edgecolor="black", label="Explicit")
    ax.barh(y, prog["n_implicit"], left=prog["n_explicit"], color="coral", edgecolor="black", label="Implicit")

    for i, total in enumerate(prog["n_total"]):
        ax.text(total + 1, i, str(total), va="center", fontsize=8)

    ax.set_yticks(list(y))
    ax.set_yticklabels(prog["label"], fontsize=8)
    ax.set_xlabel("Number of ESCO skills (explicit + implicit)")
    ax.set_title(
        f"Skills per programme — {len(prog)} programmes  "
        f"(mean {prog['n_total'].mean():.1f}, median {prog['n_total'].median():.0f})"
    )
    ax.set_xlim(0, prog["n_total"].max() * 1.08)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
