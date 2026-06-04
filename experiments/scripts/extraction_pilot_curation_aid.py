"""Gold curation aid — surface refinement candidates for the annotator.

For every gold annotation, computes structural signals and a *proposed action*
the annotator can review. Non-destructive: writes a CSV alongside the originals.

Signals computed per annotation:
  - inference_class       — implicit | explicit_with_text | explicit_inferred | explicit_unknown
  - n_co_mapped           — count of other gold URIs sharing this (doc, user_term)
  - user_term             — parsed from `notes`
  - ut_label_cosine       — cosine(user_term embedding, ESCO preferred_label embedding)
  - ut_in_esco_alt        — whether user_term matches any ESCO alt/hidden label exactly

Proposed actions (annotator reviews and accepts/overrides):
  - keep                          — annotation is well-grounded
  - downgrade_to_implicit         — explicit but user_term not in cleaned_text
  - drop_unauditable              — explicit, no user_term recorded — uncheckable
  - reconsider_uri                — user_term ↔ ESCO label cosine very low (<0.45)
  - collapse_to_concept           — one of N>1 URIs for the same user_term (concept dedup)

Output:
  experiments/validation/extraction/gold_curation.csv

Annotator workflow:
  1. Open `gold_curation.csv` in the spreadsheet of your choice.
  2. Sort by `proposed_action` to triage in groups.
  3. For each row, set `accepted_action` to keep / drop / downgrade / collapse.
  4. Re-run `extraction_pilot_gold_apply.py` (separate script) to materialise
     a refined gold from your accepted_action column.

Usage:
  python -m experiments.scripts.extraction_pilot_curation_aid
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.scraping.config import DATA_DIR

VALIDATION_DIR = DATA_DIR.parent / "experiments" / "validation" / "extraction"
DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
ESCO_PATH = DATA_DIR / "raw" / "esco" / "skills_en.csv"

PILOT_PROG_IDS = [12, 21, 25, 26, 29]
PILOT_JOB_IDS = [1, 20, 193, 253, 429]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_LOW_COSINE_THRESHOLD = 0.45

_USER_TERM_RE = re.compile(r"user\s*term\s*:\s*(.+?)(?:\s*\([^)]*\))?\s*$", re.IGNORECASE)
_WORD_BOUNDARY = re.compile(r"\W+")


def parse_user_term(notes: str) -> str:
    if not notes:
        return ""
    m = _USER_TERM_RE.search(notes.strip())
    return m.group(1).strip() if m else ""


def normalise(s: str) -> str:
    return _WORD_BOUNDARY.sub(" ", s.lower()).strip()


def load_gold(validation_dir: Path) -> pd.DataFrame:
    progs = pd.read_csv(validation_dir / "programmes_filled.csv").fillna("")
    jobs = pd.read_csv(validation_dir / "jobs_filled.csv").fillna("")
    gold = pd.concat([progs, jobs], ignore_index=True)
    gold = gold[gold["esco_uri"].str.strip().ne("")].copy()
    gold["esco_uri"] = gold["esco_uri"].str.strip()
    gold["user_term"] = gold["notes"].astype(str).apply(parse_user_term)
    gold["doc_id"] = gold["doc_id"].astype(int)
    return gold


def load_pilot_texts() -> dict[tuple[str, int], str]:
    df = pd.read_parquet(DATASET_PATH)
    progs = df[df["source_type"] == "programme"].reset_index(drop=True)
    jobs = df[df["source_type"] == "job_ad"].reset_index(drop=True)
    out: dict[tuple[str, int], str] = {}
    for pid in PILOT_PROG_IDS:
        row = progs.iloc[pid]
        out[("programme", pid)] = str(row.get("cleaned_text") or row.get("extended_description") or "")
    for jid in PILOT_JOB_IDS:
        row = jobs.iloc[jid]
        out[("job_ad", jid)] = str(row.get("cleaned_text") or row.get("description") or "")
    return out


def load_esco_alt_index(esco_path: Path) -> dict[str, set[str]]:
    """uri → set of all surface forms (lowercased) for fast lookup."""
    df = pd.read_csv(esco_path).fillna("")
    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        forms: set[str] = set()
        for col in ("preferredLabel", "altLabels", "hiddenLabels"):
            v = str(r[col]).strip()
            if not v:
                continue
            forms.update(s.strip().lower() for s in v.replace("\r", "").split("\n") if s.strip())
        out[r["conceptUri"]] = forms
    return out


def annotate(gold: pd.DataFrame, texts: dict[tuple[str, int], str], esco_alt: dict[str, set[str]]) -> pd.DataFrame:
    df = gold.copy()
    text_norm = {k: normalise(v) for k, v in texts.items()}

    # inference_class
    def _classify(row: pd.Series) -> str:
        if row["annotation_type"] == "implicit":
            return "implicit"
        ut = row["user_term"]
        if not ut:
            return "explicit_unknown"
        nt = text_norm.get((row["doc_kind"], int(row["doc_id"])), "")
        if f" {normalise(ut)} " in f" {nt} ":
            return "explicit_with_text"
        return "explicit_inferred"

    df["inference_class"] = df.apply(_classify, axis=1)

    # concept group co-mapping count
    df["concept_key"] = df["doc_kind"] + "::" + df["doc_id"].astype(str) + "::ut::" + df["user_term"]
    co_count = df.groupby("concept_key").size()
    df["n_co_mapped"] = df["concept_key"].map(co_count) - 1  # 0 = unique; >0 = duplicates
    # For rows without user_term, n_co_mapped is meaningless (each row is its own group)
    df.loc[df["user_term"] == "", "n_co_mapped"] = 0

    # user_term ↔ preferred_label cosine (semantic mismatch flag)
    logger.info("Computing user_term vs preferred_label embeddings…")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    ut_emb = model.encode(
        df["user_term"].where(df["user_term"].astype(bool), df["preferred_label"]).tolist(),
        normalize_embeddings=True, show_progress_bar=False,
    )
    pl_emb = model.encode(
        df["preferred_label"].tolist(),
        normalize_embeddings=True, show_progress_bar=False,
    )
    df["ut_label_cosine"] = np.round((ut_emb * pl_emb).sum(axis=1), 3)

    # user_term in any ESCO surface form of this URI
    def _ut_in_alt(row: pd.Series) -> bool:
        ut = row["user_term"].lower().strip()
        if not ut:
            return False
        forms = esco_alt.get(row["esco_uri"], set())
        return ut in forms

    df["ut_in_esco_alt"] = df.apply(_ut_in_alt, axis=1)

    # Proposed action
    def _propose(row: pd.Series) -> tuple[str, str]:
        ic = row["inference_class"]
        if ic == "explicit_unknown":
            return "drop_unauditable", "explicit annotation with no user_term recorded — cannot verify intent"
        if ic == "explicit_inferred":
            return "downgrade_to_implicit", "explicit but user_term not in cleaned_text — author-inferred"
        if row["ut_label_cosine"] < _LOW_COSINE_THRESHOLD and not row["ut_in_esco_alt"]:
            return "reconsider_uri", f"user_term and ESCO preferred_label semantically distant (cos={row['ut_label_cosine']:.2f})"
        if row["n_co_mapped"] > 0:
            return "collapse_to_concept", f"shares user_term with {int(row['n_co_mapped'])} other URI(s) in this doc"
        return "keep", "well-grounded"

    actions = df.apply(_propose, axis=1, result_type="expand")
    df["proposed_action"] = actions[0]
    df["proposed_reason"] = actions[1]

    # Empty column for the annotator to fill in
    df["accepted_action"] = ""

    return df


def write_outputs(df: pd.DataFrame, output_dir: Path) -> Path:
    cols = [
        "doc_kind", "doc_id", "doc_title", "esco_uri", "preferred_label",
        "annotation_type", "user_term", "inference_class",
        "n_co_mapped", "ut_label_cosine", "ut_in_esco_alt",
        "proposed_action", "proposed_reason", "accepted_action", "notes",
    ]
    out = df[cols].sort_values(["proposed_action", "doc_kind", "doc_id"]).reset_index(drop=True)
    path = output_dir / "gold_curation.csv"
    out.to_csv(path, index=False)

    # Summary report
    lines = [
        "# Gold curation aid — proposed refinements",
        "",
        f"Total annotations: **{len(df)}**",
        "",
        "## Proposed actions",
        "",
        "| action | count | meaning |",
        "| --- | ---: | --- |",
    ]
    counts = df["proposed_action"].value_counts()
    for action, n in counts.items():
        meaning = {
            "keep": "well-grounded — no change recommended",
            "downgrade_to_implicit": "term not in text → reclassify as implicit (author-inferred)",
            "drop_unauditable": "no user_term recorded → cannot verify intent",
            "reconsider_uri": "ESCO preferred_label semantically distant from author's term",
            "collapse_to_concept": "shares user_term with another URI — concept-level dedup candidate",
        }.get(action, "")
        lines.append(f"| {action} | {n} | {meaning} |")

    lines += ["", "## Per-doc breakdown of proposed actions", ""]
    pivot = df.pivot_table(
        index=["doc_kind", "doc_id"],
        columns="proposed_action", values="esco_uri", aggfunc="count", fill_value=0,
    )
    pivot["total"] = pivot.sum(axis=1)
    lines.append("| doc_kind | doc_id | " + " | ".join(pivot.columns[:-1]) + " | total |")
    lines.append("| --- | --- | " + " | ".join(["---:"] * (len(pivot.columns))) + " |")
    for (kind, did), row in pivot.iterrows():
        cells = " | ".join(str(int(row[c])) for c in pivot.columns)
        lines.append(f"| {kind} | {did} | {cells} |")

    lines += [
        "",
        "## Annotator workflow",
        "",
        "1. Open `gold_curation.csv` in your spreadsheet of choice.",
        "2. Sort by `proposed_action` to triage by class.",
        "3. For each row, fill in `accepted_action`:",
        "   - `keep` — accept as-is",
        "   - `drop` — remove from gold entirely",
        "   - `downgrade` — change `annotation_type` to `implicit`",
        "   - `relabel:<new_uri>` — replace the ESCO URI",
        "4. Save the file.",
        "5. (Optional, future) Run an apply script to materialise the refined gold.",
        "",
        "## Notes on the proposed-action rules",
        "",
        "- `drop_unauditable` and `downgrade_to_implicit` are mechanical reclassifications — "
        "no judgement on the URI itself, only on whether it's *explicit*.",
        "- `reconsider_uri` flags semantic mismatch. Many will be false alarms "
        "(jargon-heavy ESCO labels for plain user terms). Treat as a hint, not a verdict.",
        "- `collapse_to_concept` only proposes — doesn't decide which URI to keep. "
        "The concept-level rescoring (`gold_cleanup_report.md`) already handles this for metrics; "
        "physically collapsing the gold is optional.",
    ]
    report_path = output_dir / "GOLD_CURATION_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(output_dir: Path = VALIDATION_DIR) -> None:
    gold = load_gold(VALIDATION_DIR)
    texts = load_pilot_texts()
    esco_alt = load_esco_alt_index(ESCO_PATH)
    annotated = annotate(gold, texts, esco_alt)
    path = write_outputs(annotated, output_dir)
    logger.info(f"Wrote {path}")
    print()
    print(annotated["proposed_action"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=VALIDATION_DIR)
    args = parser.parse_args()
    main(output_dir=args.output_dir)
