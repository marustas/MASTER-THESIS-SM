from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_PROGRAMMES_DIR = DATA_DIR / "raw" / "programmes"
RAW_JOB_ADS_DIR = DATA_DIR / "raw" / "job_ads"

RAW_PROGRAMMES_DIR.mkdir(parents=True, exist_ok=True)
RAW_JOB_ADS_DIR.mkdir(parents=True, exist_ok=True)

# ── LAMA BPO ───────────────────────────────────────────────────────────────────
LAMA_BPO_BASE_URL = "https://lamabpo.lt"
LAMA_BPO_PROGRAMMES_URL = f"{LAMA_BPO_BASE_URL}/en/bachelors-studies/study-programmes/"

AIKOS_BASE_URL = "https://www.aikos.smm.lt"

# Field group / field keywords used to filter CS/ICT/AI programmes.
# Matched case-insensitively against the English dropdown values on the EN page.
TARGET_FIELD_GROUPS: list[str] = [
    "computer sciences",
    "computing",
    "information and communication",
]
TARGET_FIELDS: list[str] = [
    "information systems",
    "software",
    "computer",
    "cybersecurity",
    "data science",
    "artificial intelligence",
    "information technology",
]

# ── Scraping behaviour ─────────────────────────────────────────────────────────
REQUEST_DELAY: float = float(os.getenv("REQUEST_DELAY_SECONDS", "2.0"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
USER_AGENT: str = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
)

# ── Job advertisement sources ──────────────────────────────────────────────────
CVBANKAS_BASE_URL = "https://en.cvbankas.lt"
# Category IDs for IT/programming on CVbankas English site (padalinys parameter)
# 76 = IT category on en.cvbankas.lt
CVBANKAS_IT_CATEGORIES: list[int] = [76]
CVBANKAS_MAX_PAGES: int = 7

# Temporal filter: only collect ads posted within this many days
MAX_POSTING_AGE_DAYS: int = int(os.getenv("MAX_POSTING_AGE_DAYS", "90"))

# ── Playwright timeout in milliseconds ────────────────────────────────────────
PAGE_TIMEOUT: int = 30_000
NAVIGATION_TIMEOUT: int = 60_000
