"""
Step 4b — Skill quality filtering.

Removes three categories of noise from the skill extraction output:

1. Domain filter — occupation-specific / sector-specific ESCO skills whose
   labels contain no ICT-relevant keywords are discarded.  cross-sector and
   transversal skills are always kept (they are domain-agnostic competences
   such as communication, project management, critical thinking).

2. Frequency filter — skills that appear in more than FREQ_THRESHOLD of
   documents within a dataset are not discriminative and are dropped from
   every record in that dataset (IDF-style stopword removal).

3. Implicit confidence filter — implicit skills below MIN_IMPLICIT_CONF are
   dropped (raises the bar above the already-applied similarity threshold).

Input / Output:
  data/processed/programmes/programmes_with_skills.parquet  (overwritten)
  data/processed/job_ads/jobs_with_skills.parquet           (overwritten)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.scraping.config import DATA_DIR
from src.skills.esco_loader import ESCO_CSV_PATH

PROCESSED_DIR = DATA_DIR / "processed"

# ── Tunable thresholds ─────────────────────────────────────────────────────────

# Skills present in more than this fraction of documents are dropped as
# uninformative corpus-level stopwords.
FREQ_THRESHOLD: float = 0.70

# Minimum confidence kept for implicit skills (explicit confidence is set by
# the semantic similarity scorer and is typically 0.85+).
MIN_IMPLICIT_CONF: float = 0.70

# ── ICT domain keywords ────────────────────────────────────────────────────────
# Used to decide whether an occupation-specific / sector-specific ESCO skill
# is relevant to ICT/AI.  Matched case-insensitively against the skill's
# preferred label and all alternative labels.

_ICT_KEYWORDS: frozenset[str] = frozenset({
    "software", "programming", "program", "code", "coding",
    "database", "sql", "query", "nosql",
    "network", "networking", "protocol", "tcp", "http", "api",
    "cloud", "aws", "azure", "gcp", "infrastructure",
    "cyber", "security", "encryption", "firewall", "vulnerability",
    "data", "analytics", "machine learning", "deep learning",
    "artificial intelligence", "neural", "nlp", "computer vision",
    "algorithm", "computing", "computer", "ict", "information technology",
    "information system", "information security",
    "web", "frontend", "backend", "full stack", "mobile", "app",
    "devops", "ci/cd", "docker", "kubernetes", "microservice",
    "linux", "unix", "operating system", "virtualisation",
    "java", "python", "javascript", "typescript", "c++", "c#",
    "ruby", "go", "rust", "scala", "kotlin", "swift",
    "testing", "debug", "deploy", "deployment", "version control", "git",
    "architecture", "design pattern", "agile", "scrum", "sprint",
    "ux", "ui", "user interface", "user experience",
    "internet", "digital", "electronic", "hardware", "embedded",
    "robot", "automation", "script", "shell", "bash",
})

_ALWAYS_KEEP_REUSE_LEVELS: frozenset[str] = frozenset({
    "cross-sector", "transversal",
})

# Cross-sector skills whose ESCO alt labels are single generic words that
# cause systematic false positives in ICT job descriptions.
# e.g. "logistics" has "transport" as alt label → matches network/data transport
#      "energy"    has "fuel"      as alt label → matches performance/energy usage
_CROSS_SECTOR_BLOCKLIST: frozenset[str] = frozenset({
    "logistics",            # alt: "transport"
    "energy",               # alt: "fuel"
    "perform sporting activities",  # alt: "sport"
    "provide first aid",    # alt: "first aid"
    "cook",                 # alt: "cooking"
})

# ── ESCO reuse-level lookup (built once on first use) ─────────────────────────

_uri_reuse_level: dict[str, str] = {}


def _get_uri_reuse_level() -> dict[str, str]:
    global _uri_reuse_level
    if _uri_reuse_level:
        return _uri_reuse_level
    import csv
    if not ESCO_CSV_PATH.exists():
        return _uri_reuse_level
    with open(ESCO_CSV_PATH, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            uri = row.get("conceptUri", "").strip()
            level = row.get("reuseLevel", "").strip().lower()
            if uri:
                _uri_reuse_level[uri] = level
    logger.info(f"Loaded reuse levels for {len(_uri_reuse_level)} ESCO URIs")
    return _uri_reuse_level


# ── Domain filter ──────────────────────────────────────────────────────────────

def _is_ict_relevant(skill_detail: dict) -> bool:
    """
    Return True if the skill should be kept based on domain relevance.

    cross-sector / transversal skills are always kept.
    occupation-specific / sector-specific skills are kept only when the
    preferred label contains an ICT keyword.
    """
    uri = skill_detail.get("esco_uri", "")
    reuse_level = _get_uri_reuse_level().get(uri, "")

    preferred = skill_detail.get("preferred_label", "")
    if preferred in _CROSS_SECTOR_BLOCKLIST:
        return False

    if reuse_level in _ALWAYS_KEEP_REUSE_LEVELS or not reuse_level:
        return True

    return any(kw in preferred.lower() for kw in _ICT_KEYWORDS)


# ── Frequency filter ───────────────────────────────────────────────────────────

def _high_frequency_labels(records: list[list[dict]], threshold: float) -> frozenset[str]:
    """
    Return labels that appear in more than `threshold` fraction of documents.
    """
    n_docs = len(records)
    if n_docs == 0:
        return frozenset()

    doc_freq: dict[str, int] = {}
    for doc_skills in records:
        seen = set()
        for s in doc_skills:
            label = s.get("preferred_label", "")
            if label not in seen:
                doc_freq[label] = doc_freq.get(label, 0) + 1
                seen.add(label)

    cutoff = threshold * n_docs
    return frozenset(label for label, freq in doc_freq.items() if freq > cutoff)


# ── Per-record filter ──────────────────────────────────────────────────────────

def _filter_skills(
    skill_details: list[dict],
    stopwords: frozenset[str],
    min_implicit_conf: float,
) -> list[dict]:
    kept = []
    for s in skill_details:
        label = s.get("preferred_label", "")
        if label in stopwords:
            continue
        if s.get("implicit") and s.get("confidence", 1.0) < min_implicit_conf:
            continue
        if not _is_ict_relevant(s):
            continue
        kept.append(s)
    return kept


# ── Public entry point ─────────────────────────────────────────────────────────

def filter_skills_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all three quality filters to a DataFrame that has a `skill_details`
    column (list of dicts per row).  Rebuilds `explicit_skills`, `implicit_skills`,
    `all_skills`, and `skill_uris` from the filtered `skill_details`.
    """
    if "skill_details" not in df.columns:
        logger.warning("No skill_details column — skipping filter")
        return df

    records: list[list[dict]] = df["skill_details"].tolist()

    # ── Domain filter pass 1: drop non-ICT occupation-specific skills ──────────
    domain_filtered = [[s for s in doc if _is_ict_relevant(s)] for doc in records]

    # ── Frequency filter: compute stopwords on domain-filtered data ───────────
    stopwords = _high_frequency_labels(domain_filtered, FREQ_THRESHOLD)
    logger.info(
        f"Frequency stopwords (>{FREQ_THRESHOLD:.0%} of docs): "
        f"{len(stopwords)} — {sorted(stopwords)[:10]}"
    )

    # ── Apply all filters ──────────────────────────────────────────────────────
    filtered = [
        _filter_skills(doc, stopwords, MIN_IMPLICIT_CONF)
        for doc in domain_filtered
    ]

    df = df.copy()
    df["skill_details"]    = filtered
    df["explicit_skills"]  = [[s["preferred_label"] for s in doc if s.get("explicit")]  for doc in filtered]
    df["implicit_skills"]  = [[s["preferred_label"] for s in doc if s.get("implicit")]  for doc in filtered]
    df["all_skills"]       = [[s["preferred_label"] for s in doc] for doc in filtered]
    df["skill_uris"]       = [[s["esco_uri"]        for s in doc] for doc in filtered]

    before = sum(len(r) for r in records)
    after  = sum(len(r) for r in filtered)
    logger.info(
        f"Skills before: {before}  after: {after}  "
        f"({(before - after) / max(before, 1) * 100:.1f}% removed)"
    )
    return df


def run(
    programmes_path: Path = PROCESSED_DIR / "programmes" / "programmes_with_skills.parquet",
    jobs_path: Path = PROCESSED_DIR / "job_ads" / "jobs_with_skills.parquet",
) -> None:
    for path, label in [(programmes_path, "programmes"), (jobs_path, "jobs")]:
        if not path.exists():
            logger.warning(f"{label} file not found: {path}")
            continue
        logger.info(f"Filtering {label} skills: {path}")
        df = pd.read_parquet(path)
        df = filter_skills_dataframe(df)

        n = len(df)
        avg = df["all_skills"].apply(len).mean()
        zero = (df["all_skills"].apply(len) == 0).sum()
        logger.info(f"{label}: {n} records, avg {avg:.1f} skills/record, {zero} zero-skill")

        df.to_parquet(path, index=False)
        logger.info(f"Saved → {path}")


if __name__ == "__main__":
    run()
