"""Pilot — two further interventions: source-separated propagation and
ESCO parent-normalisation proxy.

Adds to the previous variant sweep:

  Same-source-only neighbour selection.  Programmes propagate skills only
  from programme neighbours; job advertisements only from job neighbours.
  Tests the hypothesis that cross-vertical embeddings contaminate the
  implicit signal even when their cosine similarity is high.

  ESCO parent-normalisation proxy.  Since the broaderConcept relation
  file is not loaded locally, cluster URIs by preferred-label token
  Jaccard ≥ 0.5 and collapse each cluster to a canonical representative
  (shortest preferred label) before scoring.  Re-tally TP/FP/FN/F1 on
  the collapsed sets.  This is a measurement-side adjustment that
  approximates rolling sibling URIs (e.g. the database trio) up to one
  parent concept on both gold and predicted sides.

The script reuses the corpus index from dataset.parquet's canonical
explicit skills and the same annotation CSVs as the previous pilot
runs.

Writes:
  experiments/validation/extraction/next_variants_metrics.csv
  experiments/validation/extraction/next_variants_per_doc.csv
  experiments/validation/extraction/NEXT_VARIANTS_REPORT.md
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
    DocumentComparison,
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


# ── Shared helpers (mirror previous sweeps) ──────────────────────────────────


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


def _build_corpus(dataset: pd.DataFrame) -> tuple[list[str], list[list[ExtractedSkill]], np.ndarray]:
    texts: list[str] = []
    explicit_skill_sets: list[list[ExtractedSkill]] = []
    source_types: list[str] = []
    for _, row in dataset.iterrows():
        text = row.get("cleaned_text") or row.get("description") or row.get("extended_description") or ""
        details = _coerce_details(row.get("skill_details"))
        explicit_only = [s for s in _details_to_extracted_skills(details) if s.explicit]
        texts.append(str(text))
        explicit_skill_sets.append(explicit_only)
        source_types.append(str(row.get("source_type", "")))
    return texts, explicit_skill_sets, np.array(source_types)


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
    score_threshold: float = 0.50
    same_source_only: bool = False


VARIANTS = [
    VariantConfig(
        name="baseline_canonical_t070",
        description="Reference: fixed K=10, τ=0.70, max-cosine scoring (no source mask)",
        scheme="fixed_k",
        tau_floor=0.70,
        fixed_k=10,
        same_source_only=False,
    ),
    VariantConfig(
        name="same_source_t070",
        description="Source-separated: fixed K=10, τ=0.70, max-cosine scoring",
        scheme="fixed_k",
        tau_floor=0.70,
        fixed_k=10,
        same_source_only=True,
    ),
    VariantConfig(
        name="best_prior_tfidf_s050",
        description="Reference: adaptive K + TF-IDF score≥0.50 (no source mask)",
        scheme="adaptive_k",
        tau_floor=0.50,
        alpha=0.85,
        k_max=25,
        use_tfidf_score=True,
        score_threshold=0.50,
        same_source_only=False,
    ),
    VariantConfig(
        name="same_source_tfidf_s050",
        description="Source-separated: adaptive K + TF-IDF score≥0.50",
        scheme="adaptive_k",
        tau_floor=0.50,
        alpha=0.85,
        k_max=25,
        use_tfidf_score=True,
        score_threshold=0.50,
        same_source_only=True,
    ),
]


# ── Per-doc extraction with optional source mask ─────────────────────────────


def _select_neighbours(sims: np.ndarray, cfg: VariantConfig) -> list[int]:
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
    raise ValueError(cfg.scheme)


def _extract_implicit_for_doc(
    extractor: ImplicitSkillExtractor,
    text: str,
    doc_idx: int,
    explicit_uris: set[str],
    cfg: VariantConfig,
    source_types: np.ndarray,
) -> list[ExtractedSkill]:
    if extractor._corpus_embeddings is None:
        raise RuntimeError("extractor not fitted")
    if not text or not text.strip():
        return []

    target_emb = extractor._corpus_embeddings[doc_idx]
    sims: np.ndarray = (extractor._corpus_embeddings @ target_emb).copy()
    sims[doc_idx] = -1.0

    if cfg.same_source_only:
        target_source = source_types[doc_idx]
        wrong_source = source_types != target_source
        sims[wrong_source] = -1.0

    neighbour_indices = _select_neighbours(sims, cfg)
    if not neighbour_indices:
        return []

    if not cfg.use_tfidf_score:
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

    accum: dict[str, float] = {}
    proto: dict[str, ExtractedSkill] = {}
    for idx in neighbour_indices:
        sim_n = float(sims[idx])
        skills_n = extractor._corpus_skill_sets[idx]
        denom = math.log(1.0 + len(skills_n)) if skills_n else 1.0
        for skill in skills_n:
            if skill.esco_uri in explicit_uris or not skill.esco_uri:
                continue
            accum[skill.esco_uri] = accum.get(skill.esco_uri, 0.0) + sim_n / denom
            proto.setdefault(skill.esco_uri, skill)

    results: list[ExtractedSkill] = []
    for uri, score in accum.items():
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
    source_types: np.ndarray,
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
            extractor, str(text), orig_idx, explicit_uris, cfg, source_types,
        )
    for jid in PILOT_JOB_IDS:
        orig_idx = int(jobs.iloc[jid]["_orig_idx"])
        row = dataset.loc[orig_idx]
        text = row.get("cleaned_text") or row.get("description") or ""
        details = _coerce_details(row.get("skill_details"))
        explicit_uris = {d["esco_uri"] for d in details if d.get("explicit") and d.get("esco_uri")}
        out[("job_ad", jid)] = _extract_implicit_for_doc(
            extractor, str(text), orig_idx, explicit_uris, cfg, source_types,
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


# ── Parent-normalisation proxy via label-token clustering ───────────────────


def _label_tokens(label: str) -> set[str]:
    return {tok for tok in label.lower().split() if len(tok) > 2}


def _build_uri_clusters(
    uri_to_label: dict[str, str],
    threshold: float = 0.5,
) -> dict[str, str]:
    """Cluster URIs by label-token Jaccard.  Returns uri -> canonical_uri.

    Greedy single-linkage clustering: walk URIs in order of shortest
    label first; for each new URI, attach to an existing cluster if
    Jaccard against the cluster's canonical exceeds threshold, else
    open a new cluster with this URI as canonical.
    """
    sorted_uris = sorted(uri_to_label.items(), key=lambda kv: (len(kv[1]), kv[1]))
    canonicals: list[tuple[str, set[str]]] = []  # (uri, tokens)
    uri_to_canonical: dict[str, str] = {}

    for uri, label in sorted_uris:
        toks = _label_tokens(label)
        if not toks:
            uri_to_canonical[uri] = uri
            continue
        best_canon: str | None = None
        best_jac = 0.0
        for canon_uri, canon_toks in canonicals:
            union = toks | canon_toks
            inter = toks & canon_toks
            jac = len(inter) / len(union) if union else 0.0
            if jac >= threshold and jac > best_jac:
                best_canon = canon_uri
                best_jac = jac
        if best_canon is not None:
            uri_to_canonical[uri] = best_canon
        else:
            uri_to_canonical[uri] = uri
            canonicals.append((uri, toks))
    return uri_to_canonical


def _collapse_comparison(
    cmps: list[DocumentComparison],
    uri_to_label: dict[str, str],
    threshold: float = 0.5,
) -> list[DocumentComparison]:
    """Apply parent normalisation to each per-doc comparison."""
    # Build a global URI->canonical mapping from the union of all URIs seen.
    all_uris: dict[str, str] = {}
    for c in cmps:
        for uri in c.gold_uris | c.extracted_uris:
            if uri not in all_uris:
                all_uris[uri] = uri_to_label.get(uri, "")
    mapping = _build_uri_clusters(all_uris, threshold=threshold)

    collapsed: list[DocumentComparison] = []
    for c in cmps:
        gold = frozenset(mapping.get(u, u) for u in c.gold_uris)
        pred = frozenset(mapping.get(u, u) for u in c.extracted_uris)
        tp = gold & pred
        fp = pred - gold
        fn = gold - pred
        collapsed.append(DocumentComparison(
            doc_kind=c.doc_kind,
            doc_id=c.doc_id,
            doc_title=c.doc_title,
            gold_uris=gold,
            extracted_uris=pred,
            tp=tp,
            fp=fp,
            fn=fn,
            near_misses={},
        ))
    return collapsed


def _build_global_uri_label_map(dataset: pd.DataFrame, annotations: pd.DataFrame) -> dict[str, str]:
    table: dict[str, str] = {}
    for _, row in dataset.iterrows():
        details = _coerce_details(row.get("skill_details"))
        for s in details:
            uri = s.get("esco_uri", "")
            if uri and uri not in table:
                table[uri] = s.get("preferred_label") or s.get("matched_text") or ""
    for r in annotations.itertuples():
        uri = str(getattr(r, "esco_uri", "")).strip()
        if uri and uri not in table:
            table[uri] = str(getattr(r, "preferred_label", "")).strip()
    return table


# ── Driver ───────────────────────────────────────────────────────────────────


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
    corpus_texts, corpus_explicit, source_types = _build_corpus(dataset)
    logger.info(f"  {len(corpus_texts)} documents in corpus")

    logger.info("Fitting ImplicitSkillExtractor (loads embedding model)…")
    dummy = ExplicitSkillExtractor.__new__(ExplicitSkillExtractor)
    extractor = ImplicitSkillExtractor(
        explicit_extractor=dummy, sim_threshold=0.0, top_k=100,
    )
    extractor.fit(texts=corpus_texts, explicit_skills_per_doc=corpus_explicit)

    uri_label_map = _build_global_uri_label_map(dataset, annotations)
    logger.info(f"URI-label map for normalisation: {len(uri_label_map)} URIs")

    sweep_rows: list[dict] = []
    per_doc_rows: list[dict] = []

    for cfg in VARIANTS:
        logger.info(f"=== Variant: {cfg.name} ===")
        logger.info(f"  {cfg.description}")
        new_implicit = _extract_pilot_implicit(extractor, dataset, cfg, source_types)
        modified_ds = _replace_pilot_details(dataset, new_implicit)
        cmps, _ = compare_sample(annotations, modified_ds)

        # Raw (un-normalised) metrics
        agg = aggregate_metrics(cmps)
        micro = agg[agg["doc_kind"] == "ALL"].iloc[0]
        sweep_rows.append({
            "variant": cfg.name,
            "normalisation": "none",
            "description": cfg.description,
            "tp": int(micro["tp"]),
            "fp": int(micro["fp"]),
            "fn": int(micro["fn"]),
            "near_miss": int(micro["near_miss"]),
            "precision": float(micro["precision"]),
            "recall": float(micro["recall"]),
            "f1": float(micro["f1"]),
        })

        # Normalised (parent-cluster proxy) metrics
        collapsed = _collapse_comparison(cmps, uri_label_map, threshold=0.5)
        agg_collapsed = aggregate_metrics(collapsed)
        if not agg_collapsed.empty:
            micro_c = agg_collapsed[agg_collapsed["doc_kind"] == "ALL"].iloc[0]
            sweep_rows.append({
                "variant": cfg.name,
                "normalisation": "label_cluster_j0.5",
                "description": cfg.description,
                "tp": int(micro_c["tp"]),
                "fp": int(micro_c["fp"]),
                "fn": int(micro_c["fn"]),
                "near_miss": 0,
                "precision": float(micro_c["precision"]),
                "recall": float(micro_c["recall"]),
                "f1": float(micro_c["f1"]),
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
    sweep_df.to_csv(output_dir / "next_variants_metrics.csv", index=False)
    per_doc_df.to_csv(output_dir / "next_variants_per_doc.csv", index=False)

    lines: list[str] = ["# Pilot — source-separated propagation + parent-normalisation proxy", ""]
    lines.append("Micro-averaged metrics on the 10 pilot documents at each variant, both with raw URIs and with label-cluster normalisation (proxy for ESCO `broaderConcept` rollup):")
    lines.append("")
    lines.append("| Variant | Normalisation | TP | FP | FN | NM | Precision | Recall | F1 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sweep_rows:
        lines.append(
            f"| `{r['variant']}` | {r['normalisation']} | {r['tp']} | {r['fp']} | "
            f"{r['fn']} | {r['near_miss']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    lines.append("")
    lines.append("Variant descriptions:")
    lines.append("")
    for cfg in VARIANTS:
        lines.append(f"- `{cfg.name}` — {cfg.description}")
    lines.append("")
    lines.append("Normalisation key:")
    lines.append("")
    lines.append("- `none` — URIs compared as-is (canonical TP/FP/FN definitions)")
    lines.append("- `label_cluster_j0.5` — proxy for ESCO parent rollup: greedy single-linkage clustering of all URIs (gold ∪ predicted) by preferred-label token Jaccard ≥ 0.5; canonical per cluster = shortest label. Both sides collapsed before scoring.")
    (output_dir / "NEXT_VARIANTS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(f"Metrics → {output_dir / 'next_variants_metrics.csv'}")
    logger.info(f"Per-doc → {output_dir / 'next_variants_per_doc.csv'}")
    logger.info(f"Report → {output_dir / 'NEXT_VARIANTS_REPORT.md'}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(output_dir=args.output_dir)
