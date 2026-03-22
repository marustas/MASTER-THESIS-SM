"""
ESCO skills taxonomy loader and index builder.

Supports two data sources (in priority order):
  1. Local CSV download — fast, offline, preferred for batch processing.
     Download ESCO v1.2.1 from https://esco.ec.europa.eu/en/use-esco/download
     and place the skills CSV at: data/raw/esco/skills_en.csv
  2. ESCO REST API — used as fallback or for single-skill lookups.
     Base URL: https://ec.europa.eu/esco/api

The loader builds two in-memory indices:
  - label_index:  lowercased label → EscoSkill
  - uri_index:    conceptUri       → EscoSkill

These are used by the explicit extractor for fast phrase matching.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import httpx
from loguru import logger

from src.scraping.config import DATA_DIR

ESCO_CSV_PATH = DATA_DIR / "raw" / "esco" / "skills_en.csv"
ESCO_API_BASE = "https://ec.europa.eu/esco/api"
ESCO_API_VERSION = "v1"
ESCO_LANGUAGE = "en"
API_RATE_LIMIT_DELAY = 0.3  # seconds between API requests


@dataclass
class EscoSkill:
    """A single ESCO skill concept."""

    uri: str
    preferred_label: str
    alt_labels: list[str] = field(default_factory=list)
    skill_type: Optional[str] = None   # "skill/competence" | "knowledge"
    reuse_level: Optional[str] = None  # "cross-sector" | "occupation-specific" | etc.
    description: Optional[str] = None

    @property
    def all_labels(self) -> list[str]:
        """All surface forms: preferred + alternative labels."""
        return [self.preferred_label] + self.alt_labels


@dataclass
class EscoIndex:
    """In-memory index over ESCO skills for fast lookup."""

    skills: list[EscoSkill] = field(default_factory=list)
    label_index: dict[str, EscoSkill] = field(default_factory=dict)  # lower label → skill
    uri_index: dict[str, EscoSkill] = field(default_factory=dict)    # uri → skill

    def build(self) -> None:
        """Build lookup indices from loaded skills."""
        self.label_index.clear()
        self.uri_index.clear()
        for skill in self.skills:
            self.uri_index[skill.uri] = skill
            for label in skill.all_labels:
                key = label.lower().strip()
                if key and key not in self.label_index:
                    self.label_index[key] = skill
        logger.info(
            f"ESCO index built: {len(self.skills)} skills, "
            f"{len(self.label_index)} label entries"
        )

    def lookup_label(self, label: str) -> Optional[EscoSkill]:
        return self.label_index.get(label.lower().strip())

    def lookup_uri(self, uri: str) -> Optional[EscoSkill]:
        return self.uri_index.get(uri)

    def __len__(self) -> int:
        return len(self.skills)


# ── CSV loader ─────────────────────────────────────────────────────────────────

def load_from_csv(csv_path: Path = ESCO_CSV_PATH) -> EscoIndex:
    """
    Load ESCO skills from the official CSV download.

    Expected columns (ESCO v1.2.1 skills_en.csv):
      conceptUri, skillType, reuseLevel, preferredLabel, altLabels,
      hiddenLabels, status, modifiedDate, scopeNote, definition,
      inScheme, description, code
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ESCO CSV not found at {csv_path}.\n"
            "Download ESCO v1.2.1 from https://esco.ec.europa.eu/en/use-esco/download "
            "and place skills_en.csv in data/raw/esco/"
        )

    skills: list[EscoSkill] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uri = row.get("conceptUri", "").strip()
            preferred = row.get("preferredLabel", "").strip()
            if not uri or not preferred:
                continue

            # altLabels in ESCO CSV are newline-separated within the cell
            alt_raw = row.get("altLabels", "") or ""
            alt_labels = [
                lbl.strip()
                for lbl in alt_raw.replace("\\n", "\n").splitlines()
                if lbl.strip() and lbl.strip() != preferred
            ]

            skills.append(EscoSkill(
                uri=uri,
                preferred_label=preferred,
                alt_labels=alt_labels,
                skill_type=row.get("skillType", "").strip() or None,
                reuse_level=row.get("reuseLevel", "").strip() or None,
                description=(row.get("description") or row.get("definition") or "").strip() or None,
            ))

    logger.info(f"Loaded {len(skills)} skills from {csv_path}")
    index = EscoIndex(skills=skills)
    index.build()
    return index


# ── API loader (fallback / supplemental) ──────────────────────────────────────

class EscoApiClient:
    """Thin wrapper around the ESCO REST API v1."""

    def __init__(self, base_url: str = ESCO_API_BASE, delay: float = API_RATE_LIMIT_DELAY):
        self._base = base_url
        self._delay = delay
        self._client = httpx.Client(timeout=15.0)

    def search_skills(self, text: str, limit: int = 10) -> list[EscoSkill]:
        """Search for skills matching a free-text query."""
        url = f"{self._base}/search"
        params = {
            "text": text,
            "type": "skill",
            "language": ESCO_LANGUAGE,
            "limit": limit,
        }
        try:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            skills = []
            for item in data.get("_embedded", {}).get("results", []):
                skills.append(self._parse_api_result(item))
            time.sleep(self._delay)
            return skills
        except Exception as exc:
            logger.warning(f"ESCO API search failed for '{text}': {exc}")
            return []

    def get_skill(self, uri: str) -> Optional[EscoSkill]:
        """Fetch a single skill by its URI."""
        url = f"{self._base}/resource/skill"
        try:
            resp = self._client.get(url, params={"uri": uri, "language": ESCO_LANGUAGE})
            resp.raise_for_status()
            return self._parse_api_result(resp.json())
        except Exception as exc:
            logger.warning(f"ESCO API get skill failed for '{uri}': {exc}")
            return None

    @staticmethod
    def _parse_api_result(data: dict) -> EscoSkill:
        uri = data.get("uri", "")
        preferred = data.get("title", "") or data.get("preferredLabel", {}).get("en", "")
        alt_labels = list(data.get("alternativeLabel", {}).get(ESCO_LANGUAGE, []))
        return EscoSkill(
            uri=uri,
            preferred_label=preferred,
            alt_labels=alt_labels,
            skill_type=data.get("skillType"),
            description=data.get("description", {}).get(ESCO_LANGUAGE, {}).get("literal"),
        )

    def close(self) -> None:
        self._client.close()


# ── Convenience loader ─────────────────────────────────────────────────────────

def load_esco_index(csv_path: Path = ESCO_CSV_PATH) -> EscoIndex:
    """
    Load ESCO index from CSV if available, otherwise raise with instructions.
    Use `EscoApiClient` for on-demand API queries.
    """
    return load_from_csv(csv_path)
