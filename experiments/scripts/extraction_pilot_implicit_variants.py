"""Pilot variant comparison — adaptive top-K and TF-IDF-style implicit scoring.

Tests two structural changes to the implicit extractor against the canonical
fixed-K / max-cosine baseline, on the same 10 pilot documents:

  Variant A — adaptive top-K.  Replace the fixed top_k=10 + threshold scheme
    with: keep neighbours whose cosine >= max(tau_floor, alpha * cos_max),
    capped at K_max.  alpha=0.85, K_max=25, tau_floor=0.50.

  Variant B — TF-IDF-style multi-neighbour scoring.  Same neighbour selection
    as Variant A.  For each candidate URI u, compute
        score(u) = sum over neighbours n containing u  of
                       sim(n) / log(1 + |skills(n)|)
    Keep URIs whose score exceeds a configurable threshold.  Reported
    confidence = clip(score, 0, 1) for use by downstream filters.

The script reuses the corpus index built from dataset.parquet's canonical
explicit skills, then injects new skill_details for the 10 pilot docs and
runs compare_sample against the same annotation CSVs.

Writes:
  experiments/validation/extraction/variant_sweep_metrics.csv
  experiments/validation/extraction/variant_sweep_per_doc.csv
  experiments/validation/extraction/VARIANT_SWEEP_REPORT.md

Usage:
  python -m experiments.scripts.extraction_pilot_implicit_variants
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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


# ── Shared helpers (mirror threshold_sweep) ──────────────────────────────────


def _details_to_extracted_skills(details: list[dict]) -> list[ExtractedSkill]:
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
    texts: list[str] = []
    explicit_skill_sets: list[list[ExtractedSkill]] = []
    for _, row in dataset.iterrows():
        text = row.get("cleaned_text") or row.get("description") or row.get("extended_description") or ""
        details = _coerce_details(row.get("skill_details"))
        explicit_only = [s for s in _details_to_extracted_skills(details) if s.explicit]
        texts.append(str(text))
        explicit_skill_sets.append(explicit_only)
    return texts, explicit_skill_sets


# ── Variant configuration ────────────────────────────────────────────────────


@dataclass(frozen=True)
class VariantConfig:
    name: str
    description: str
    scheme: str   # "fixed_k" | "adaptive_k"
    tau_floor: float = 0.50
    fixed_k: int = 10
    alpha: float = 0.85
    k_max: int = 25
    use_tfidf_score: bool = False
    score_threshold: float = 0.30


VARIANTS = [
    VariantConfig(
        name="baseline_fixedK_t070",
        description="Fixed K=10, τ=0.70 (canonical baseline neighbour selection)",
        scheme="fixed_k",
        tau_floor=0.70,
        fixed_k=10,
    ),
    VariantConfig(
        name="baseline_fixedK_t050",
        description="Fixed K=10, τ=0.50 (relaxed-threshold baseline from prior sweep)",
        scheme="fixed_k",
        tau_floor=0.50,
        fixed_k=10,
    ),
    VariantConfig(
        name="adaptive_k",
        description="Adaptive K: τ_eff = max(0.50, 0.85·cos_max), K_max=25, max-cosine scoring",
        scheme="adaptive_k",
        tau_floor=0.50,
        alpha=0.85,
        k_max=25,
        use_tfidf_score=False,
    ),
    VariantConfig(
        name="adaptive_k_tfidf_s030",
        description="Adaptive K + TF-IDF score = Σ sim/log(1+|skills|), keep score≥0.30",
        scheme="adaptive_k",
        tau_floor=0.50,
        alpha=0.85,
        k_max=25,
        use_tfidf_score=True,
        score_threshold=0.30,
    ),
    VariantConfig(
        name="adaptive_k_tfidf_s050",
        description="Same as above but score_threshold=0.50 (more conservative)",
        scheme="adaptive_k",
        tau_floor=0.50,
        alpha=0.85,
        k_max=25,
        use_tfidf_score=True,
        score_threshold=0.50,
    ),
]


# ── Per-variant implicit extraction ──────────────────────────────────────────


def _select_neighbours(
    sims: np.ndarray,
    cfg: VariantConfig,
) -> list[int]:
    """Return neighbour indices according to the variant's selection rule."""
    ranked = np.argsort(sims)[::-1]
    if cfg.scheme == "fixed_k":
        return [int(i) for i in ranked if sims[i] >= cfg.tau_floor][: cfg.fixed_k]
    if cfg.scheme == "adaptive_k":
        if sims.size == 0:
            return []
        cos_max = float(sims[ranked[0]])
        if cos_max < cfg.tau_floor:
            return []
        adaptive_floor = max(cfg.tau_floor, cfg.alpha * cos_max)
        return [int(i) for i in ranked if sims[i] >= adaptive_floor][: cfg.k_max]
    raise ValueError(f"unknown scheme {cfg.scheme!r}")


def _extract_implicit_for_doc(
    extractor: ImplicitSkillExtractor,
    text: str,
    doc_idx: int,
    explicit_uris: set[str],
    cfg: VariantConfig,
) -> list[ExtractedSkill]:
    """Re-implement extract() with variant-specific neighbour selection + scoring."""
    if extractor._corpus_embeddings is None:
        raise RuntimeError("extractor not fitted")
    if not text or not text.strip():
        return []

    target_emb = extractor._corpus_embeddings[doc_idx]
    sims: np.ndarray = extractor._corpus_embeddings @ target_emb
    sims = sims.copy()
    sims[doc_idx] = -1.0

    neighbour_indices = _select_neighbours(sims, cfg)
    if not neighbour_indices:
        return []

    if not cfg.use_tfidf_score:
        # Max-cosine scoring (current behaviour).
        candidates: dict[str, tuple[ExtractedSkill, float]] = {}
        for idx in neighbour_indices:
            sim_n = float(sims[idx])
            for skill in extractor._corpus_skill_sets[idx]:
                if skill.esco_uri in explicit_uris or not skill.esco_uri:
                    continue
                if skill.esco_uri not in candidates or sim_n > candidates[skill.esco_uri][1]:
                    candidates[skill.esco_uri] = (skill, sim_n)
        return [
            ExtractedSkill(
                esco_uri=uri,
                preferred_label=skill.preferred_label,
                matched_text=skill.matched_text,
                explicit=False,
                implicit=True,
                confidence=sim,
            )
            for uri, (skill, sim) in candidates.items()
        ]

    # TF-IDF-style scoring.
    accum_score: dict[str, float] = {}
    proto: dict[str, ExtractedSkill] = {}
    for idx in neighbour_indices:
        sim_n = float(sims[idx])
        skills_n = extractor._corpus_skill_sets[idx]
        denom = math.log(1.0 + len(skills_n)) if skills_n else 1.0
        for skill in skills_n:
            if skill.esco_uri in explicit_uris or not skill.esco_uri:
                continue
            accum_score[skill.esco_uri] = accum_score.get(skill.esco_uri, 0.0) + sim_n / denom
            proto.setdefault(skill.esco_uri, skill)

    results: list[ExtractedSkill] = []
    for uri, score in accum_score.items():
        if score < cfg.score_threshold:
            continue
        skill = proto[uri]
        results.append(ExtractedSkill(
            esco_uri=uri,
            preferred_label=skill.preferred_label,
            matched_text=skill.matched_text,
            explicit=False,
            implicit=True,
            confidence=float(min(score, 1.0)),
        ))
    return results


def _extract_pilot_implicit(
    extractor: ImplicitSkillExtractor,
    dataset: pd.DataFrame,
    cfg: VariantConfig,
) -> dict[tuple[str, int], list[ExtractedSkill]]:
    progs = dataset[dataset["source_type"] == "programme"].reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    jobs = dataset[dataset["source_type"] == "job_ad"].reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    out: dict[tuple[str, int], list[ExtractedSkill]] = {}

    for pid in PILOT_PROG_IDS:
        orig_idx = int(progs.iloc[pid]["_orig_idx"])
        row = dataset.loc[orig_idx]
        text = row.get("cleaned_text") or row.get("extended_description") or ""
        details = _coerce_details(row.get("skill_details"))
        explicit_uris = {d["esco_uri"] for d in details if d.get("explicit") and d.get("esco_uri")}
        out[("programme", pid)] = _extract_implicit_for_doc(
            extractor, str(text), orig_idx, explicit_uris, cfg,
        )

    for jid in PILOT_JOB_IDS:
        orig_idx = int(jobs.iloc[jid]["_orig_idx"])
        row = dataset.loc[orig_idx]
        text = row.get("cleaned_text") or row.get("description") or ""
        details = _coerce_details(row.get("skill_details"))
        explicit_uris = {d["esco_uri"] for d in details if d.get("explicit") and d.get("esco_uri")}
        out[("job_ad", jid)] = _extract_implicit_for_doc(
            extractor, str(text), orig_idx, explicit_uris, cfg,
        )

    return out


def _replace_pilot_details(
    dataset: pd.DataFrame,
    new_implicit: dict[tuple[str, int], list[ExtractedSkill]],
) -> pd.DataFrame:
    df = dataset.copy()
    progs = df[df["source_type"] == "programme"].reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    jobs = df[df["source_type"] == "job_ad"].reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    for pid in PILOT_PROG_IDS:
        orig_idx = int(progs.iloc[pid]["_orig_idx"])
        details = _coerce_details(df.at[orig_idx, "skill_details"])
        explicit_dicts = [d for d in details if d.get("explicit")]
        df.at[orig_idx, "skill_details"] = explicit_dicts + [
            _skill_to_dict(s) for s in new_implicit.get(("programme", pid), [])
        ]
    for jid in PILOT_JOB_IDS:
        orig_idx = int(jobs.iloc[jid]["_orig_idx"])
        details = _coerce_details(df.at[orig_idx, "skill_details"])
        explicit_dicts = [d for d in details if d.get("explicit")]
        df.at[orig_idx, "skill_details"] = explicit_dicts + [
            _skill_to_dict(s) for s in new_implicit.get(("job_ad", jid), [])
        ]
    return df


def main(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {DATASET_PATH}…")
    dataset = pd.read_parquet(DATASET_PATH)
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

    logger.info("Fitting ImplicitSkillExtractor (loads embedding model)…")
    dummy_explicit_extractor = ExplicitSkillExtractor.__new__(ExplicitSkillExtractor)
    extractor = ImplicitSkillExtractor(
        explicit_extractor=dummy_explicit_extractor,
        sim_threshold=0.0,  # selection is done per-variant below
        top_k=100,
    )
    extractor.fit(texts=corpus_texts, explicit_skills_per_doc=corpus_explicit)

    sweep_rows: list[dict] = []
    per_doc_rows: list[dict] = []

    for cfg in VARIANTS:
        logger.info(f"=== Variant: {cfg.name} ===")
        logger.info(f"  {cfg.description}")
        new_implicit = _extract_pilot_implicit(extractor, dataset, cfg)

        counts = {(k, d): len(v) for (k, d), v in new_implicit.items()}
        logger.info(f"  per-doc implicit count: {counts}")

        modified_ds = _replace_pilot_details(dataset, new_implicit)
        cmps, _ = compare_sample(annotations, modified_ds)
        agg = aggregate_metrics(cmps)
        micro = agg[agg["doc_kind"] == "ALL"].iloc[0]

        sweep_rows.append({
            "variant": cfg.name,
            "description": cfg.description,
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
            per_doc_rows.append({
                "variant": cfg.name,
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
    per_doc_df = pd.DataFrame(per_doc_rows)

    sweep_df.to_csv(output_dir / "variant_sweep_metrics.csv", index=False)
    per_doc_df.to_csv(output_dir / "variant_sweep_per_doc.csv", index=False)

    lines: list[str] = ["# Implicit-extractor variant comparison — pilot results", ""]
    lines.append("Micro-averaged metrics on the 10 pilot documents:")
    lines.append("")
    lines.append("| Variant | TP | FP | FN | NM | Precision | Recall | F1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sweep_rows:
        lines.append(
            f"| `{r['variant']}` | {r['tp']} | {r['fp']} | {r['fn']} | {r['near_miss']} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    lines.append("")
    lines.append("Variant descriptions:")
    lines.append("")
    for cfg in VARIANTS:
        lines.append(f"- `{cfg.name}` — {cfg.description}")
    (output_dir / "VARIANT_SWEEP_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(f"Metrics → {output_dir / 'variant_sweep_metrics.csv'}")
    logger.info(f"Per-doc → {output_dir / 'variant_sweep_per_doc.csv'}")
    logger.info(f"Report → {output_dir / 'VARIANT_SWEEP_REPORT.md'}")
    logger.info("Variant sweep complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(output_dir=args.output_dir)
