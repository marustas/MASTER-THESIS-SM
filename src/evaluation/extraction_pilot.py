"""Skill-extraction pilot validation (intermediate-defence scale).

Supports a small-N manual validation of the explicit + implicit ESCO
skill extractor:

  1. Stratified sampling of programmes and job ads (one row per cluster,
     deterministic given a seed).
  2. Empty annotation-template generator (CSV-per-document) for the
     annotator to fill in blind.
  3. Comparison of manual annotations against the extractor's output,
     producing per-document precision / recall / F1, set Jaccard, and a
     near-miss flag for false-positive URIs whose preferred-label tokens
     overlap heavily with a gold URI.

Tooling is deliberately offline: ESCO labels are looked up from the
local ESCO CSV via ``src.skills.esco_loader``.  No network calls.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from src.scraping.config import DATA_DIR

# ── Paths ────────────────────────────────────────────────────────────────────

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"

# ── Sampling ─────────────────────────────────────────────────────────────────

DEFAULT_N = 5
DEFAULT_SEED = 42
NEAR_MISS_JACCARD = 0.5


def _stratified_pick(
    df: pd.DataFrame,
    *,
    n: int,
    seed: int,
    cluster_col: str = "cluster_label",
) -> pd.DataFrame:
    """Pick ``n`` rows, one per cluster, deterministic in ``seed``.

    Clusters are visited in descending size order so heavily populated
    clusters are exercised first.  If fewer than ``n`` distinct clusters
    exist the remainder is filled by drawing additional rows from the
    largest clusters; if more than ``n`` clusters exist the smallest
    ones are skipped.  Noise rows (cluster < 0 or NaN) are excluded.
    """
    clean = df[df[cluster_col].notna() & (df[cluster_col] >= 0)].copy()
    sizes = clean[cluster_col].value_counts().sort_values(ascending=False)

    picked: list[pd.Series] = []
    for cluster, _ in sizes.items():
        if len(picked) >= n:
            break
        candidates = clean[clean[cluster_col] == cluster]
        picked.append(candidates.sample(n=1, random_state=seed + int(cluster)).iloc[0])

    if len(picked) < n:
        used_ids = {id(s) for s in picked}
        remainder = clean.sample(
            n=n - len(picked), random_state=seed + 999
        )
        for _, row in remainder.iterrows():
            if id(row) in used_ids:
                continue
            picked.append(row)

    out = pd.DataFrame(picked).reset_index(drop=True)
    return out


def stratified_sample_programmes(
    dataset: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return ``n`` programme rows stratified by ``cluster_label``."""
    progs = dataset[dataset["source_type"] == "programme"].copy()
    progs = progs.reset_index(drop=True)
    progs.insert(0, "programme_id", progs.index.astype(int))
    picked = _stratified_pick(progs, n=n, seed=seed)
    cols = [
        "programme_id",
        "name",
        "institution",
        "cluster_label",
        "extended_description",
        "skill_uris",
        "skill_details",
    ]
    cols = [c for c in cols if c in picked.columns]
    return picked.loc[:, cols].reset_index(drop=True)


def stratified_sample_jobs(
    dataset: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return ``n`` job-ad rows stratified by ``cluster_label``."""
    jobs = dataset[dataset["source_type"] == "job_ad"].copy()
    jobs = jobs.reset_index(drop=True)
    jobs.insert(0, "job_id", jobs.index.astype(int))
    picked = _stratified_pick(jobs, n=n, seed=seed)
    cols = [
        "job_id",
        "job_title",
        "employer_sector",
        "cluster_label",
        "description",
        "skill_uris",
        "skill_details",
    ]
    cols = [c for c in cols if c in picked.columns]
    return picked.loc[:, cols].reset_index(drop=True)


# ── Annotation template ──────────────────────────────────────────────────────


def _coerce_details(value) -> list[dict]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def write_annotation_template(
    sample: pd.DataFrame,
    output_path: Path,
    *,
    doc_kind: str,
) -> Path:
    """Write a blank annotation CSV the annotator fills in.

    One row per (document, blank ESCO URI slot).  Eight pre-filled blank
    slots per document; the annotator deletes unused rows and appends
    more if needed.  Does *not* leak the extractor's output, so the
    annotator can record their independent reading first.
    """
    if doc_kind == "programme":
        id_col, title_col = "programme_id", "name"
        kind_label = "programme"
    elif doc_kind == "job_ad":
        id_col, title_col = "job_id", "job_title"
        kind_label = "job_ad"
    else:
        raise ValueError(f"doc_kind must be 'programme' or 'job_ad', got {doc_kind!r}")

    rows: list[dict] = []
    for _, doc in sample.iterrows():
        for _ in range(8):
            rows.append(
                {
                    "doc_kind": kind_label,
                    "doc_id": int(doc[id_col]),
                    "doc_title": doc[title_col],
                    "esco_uri": "",
                    "preferred_label": "",
                    "annotation_type": "",  # explicit | implicit
                    "annotator_confidence": "",  # high | medium | low
                    "notes": "",
                }
            )

    template = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    logger.info(f"Annotation template → {output_path}  ({len(template)} blank rows)")
    return output_path


# ── Comparison ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DocumentComparison:
    """Per-document comparison result."""

    doc_kind: str
    doc_id: int
    doc_title: str
    gold_uris: frozenset[str]
    extracted_uris: frozenset[str]
    tp: frozenset[str]
    fp: frozenset[str]
    fn: frozenset[str]
    near_misses: dict[str, tuple[str, float]] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        denom = len(self.tp) + len(self.fp)
        return len(self.tp) / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = len(self.tp) + len(self.fn)
        return len(self.tp) / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def jaccard(self) -> float:
        union = self.gold_uris | self.extracted_uris
        return len(self.gold_uris & self.extracted_uris) / len(union) if union else 0.0


def _label_tokens(label: str) -> set[str]:
    return {tok for tok in label.lower().split() if len(tok) > 2}


def _near_miss(
    fp_uri: str,
    fp_label: str,
    gold_uri_labels: dict[str, str],
    threshold: float = NEAR_MISS_JACCARD,
) -> tuple[str, float] | None:
    fp_tokens = _label_tokens(fp_label)
    if not fp_tokens:
        return None
    best: tuple[str, float] | None = None
    for g_uri, g_label in gold_uri_labels.items():
        g_tokens = _label_tokens(g_label)
        if not g_tokens:
            continue
        union = fp_tokens | g_tokens
        inter = fp_tokens & g_tokens
        score = len(inter) / len(union) if union else 0.0
        if score >= threshold and (best is None or score > best[1]):
            best = (g_uri, score)
    return best


def compare_document(
    *,
    doc_kind: str,
    doc_id: int,
    doc_title: str,
    gold_uris: Iterable[str],
    extracted: list[dict],
    uri_to_label: dict[str, str] | None = None,
) -> DocumentComparison:
    """Compare one document's gold URIs against extractor output."""
    extracted_uri_label = {
        s["esco_uri"]: s.get("preferred_label") or s.get("matched_text") or ""
        for s in extracted
        if s.get("esco_uri")
    }
    extracted_set = frozenset(extracted_uri_label)
    gold_set = frozenset(gold_uris)
    tp = gold_set & extracted_set
    fp = extracted_set - gold_set
    fn = gold_set - extracted_set

    label_index = dict(uri_to_label or {})
    for u, lab in extracted_uri_label.items():
        label_index.setdefault(u, lab)
    gold_labels = {u: label_index.get(u, "") for u in gold_set}

    near: dict[str, tuple[str, float]] = {}
    for fp_uri in fp:
        hit = _near_miss(fp_uri, label_index.get(fp_uri, ""), gold_labels)
        if hit is not None:
            near[fp_uri] = hit

    return DocumentComparison(
        doc_kind=doc_kind,
        doc_id=doc_id,
        doc_title=doc_title,
        gold_uris=gold_set,
        extracted_uris=extracted_set,
        tp=tp,
        fp=fp,
        fn=fn,
        near_misses=near,
    )


def compare_sample(
    annotations: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    uri_to_label: dict[str, str] | None = None,
) -> tuple[list[DocumentComparison], pd.DataFrame]:
    """Compare a filled-in annotation CSV against the dataset's extractor output.

    Returns a list of per-document comparison records and a long-form
    diff DataFrame (one row per URI per document, labelled TP / FP / FN
    / NEAR_MISS) suitable for inclusion in the report markdown.
    """
    if "doc_kind" not in annotations.columns:
        raise ValueError("annotations CSV missing required column 'doc_kind'")

    progs = dataset[dataset["source_type"] == "programme"].reset_index(drop=True)
    progs.insert(0, "programme_id", progs.index.astype(int))
    jobs = dataset[dataset["source_type"] == "job_ad"].reset_index(drop=True)
    jobs.insert(0, "job_id", jobs.index.astype(int))

    valid = annotations[annotations["esco_uri"].fillna("").str.strip() != ""].copy()
    grouped = valid.groupby(["doc_kind", "doc_id", "doc_title"], sort=False)

    cmps: list[DocumentComparison] = []
    diff_rows: list[dict] = []

    for (kind, doc_id, title), block in grouped:
        gold_uris = block["esco_uri"].str.strip().tolist()
        if kind == "programme":
            row = progs[progs["programme_id"] == int(doc_id)]
        else:
            row = jobs[jobs["job_id"] == int(doc_id)]
        if row.empty:
            logger.warning(f"{kind} doc_id {doc_id} not found in dataset, skipping")
            continue

        details = _coerce_details(row.iloc[0].get("skill_details"))
        annotation_labels = {
            r["esco_uri"].strip(): (r.get("preferred_label") or "").strip()
            for _, r in block.iterrows()
        }
        merged_uri_label = dict(uri_to_label or {})
        merged_uri_label.update({u: lab for u, lab in annotation_labels.items() if lab})

        cmp = compare_document(
            doc_kind=kind,
            doc_id=int(doc_id),
            doc_title=str(title),
            gold_uris=gold_uris,
            extracted=details,
            uri_to_label=merged_uri_label,
        )
        cmps.append(cmp)

        for uri in cmp.tp:
            diff_rows.append({"doc_kind": kind, "doc_id": int(doc_id), "esco_uri": uri,
                              "label": merged_uri_label.get(uri, ""), "verdict": "TP"})
        for uri in cmp.fp:
            verdict = "NEAR_MISS" if uri in cmp.near_misses else "FP"
            note = ""
            if uri in cmp.near_misses:
                g_uri, score = cmp.near_misses[uri]
                note = f"matches gold {g_uri} (label Jaccard {score:.2f})"
            diff_rows.append({"doc_kind": kind, "doc_id": int(doc_id), "esco_uri": uri,
                              "label": merged_uri_label.get(uri, ""), "verdict": verdict,
                              "note": note})
        for uri in cmp.fn:
            diff_rows.append({"doc_kind": kind, "doc_id": int(doc_id), "esco_uri": uri,
                              "label": merged_uri_label.get(uri, ""), "verdict": "FN"})

    diff_df = pd.DataFrame(diff_rows)
    return cmps, diff_df


# ── Metric aggregation ───────────────────────────────────────────────────────


def aggregate_metrics(cmps: list[DocumentComparison]) -> pd.DataFrame:
    """Per-document metrics + micro-averaged totals as a DataFrame."""
    rows = []
    for c in cmps:
        rows.append(
            {
                "doc_kind": c.doc_kind,
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "gold_n": len(c.gold_uris),
                "extracted_n": len(c.extracted_uris),
                "tp": len(c.tp),
                "fp": len(c.fp),
                "fn": len(c.fn),
                "near_miss": len(c.near_misses),
                "precision": round(c.precision, 4),
                "recall": round(c.recall, 4),
                "f1": round(c.f1, 4),
                "jaccard": round(c.jaccard, 4),
            }
        )
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    total_tp = int(df["tp"].sum())
    total_fp = int(df["fp"].sum())
    total_fn = int(df["fn"].sum())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    overall = {
        "doc_kind": "ALL",
        "doc_id": -1,
        "doc_title": "micro-average",
        "gold_n": int(df["gold_n"].sum()),
        "extracted_n": int(df["extracted_n"].sum()),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "near_miss": int(df["near_miss"].sum()),
        "precision": round(micro_p, 4),
        "recall": round(micro_r, 4),
        "f1": round(micro_f1, 4),
        "jaccard": round(df["jaccard"].mean(), 4),
    }
    return pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
