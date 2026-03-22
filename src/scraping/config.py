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
LAMA_BPO_BASE_URL = "https://www.lamabpo.lt"
LAMA_BPO_PROGRAMMES_URL = (
    f"{LAMA_BPO_BASE_URL}/pirmosios-pakopos-ir-vientisosios-studijos/programu-sarasas/"
)

# Field group / field keywords used to filter CS/ICT/AI programmes.
# These are matched case-insensitively against the Lithuanian dropdown values.
TARGET_FIELD_GROUPS: list[str] = [
    "informatika",
    "informacinės technologijos",
    "kompiuterija",
    "dirbtinis intelektas",
]
TARGET_FIELDS: list[str] = [
    "informatika",
    "programų sistemos",
    "informacinės sistemos",
    "kompiuterių inžinerija",
    "kibernetinis saugumas",
    "duomenų mokslas",
    "dirbtinis intelektas",
    "informacinės technologijos",
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
CVBANKAS_BASE_URL = "https://www.cvbankas.lt"
# Category IDs for IT/programming on CVbankas (padalinys parameter)
# 2 = Informacinės technologijos (IT), confirmed from site category structure
CVBANKAS_IT_CATEGORIES: list[int] = [2]
CVBANKAS_MAX_PAGES: int = int(os.getenv("CVBANKAS_MAX_PAGES", "50"))

LINKEDIN_JOBS_URL = "https://www.linkedin.com/jobs/search/"
# LinkedIn geoId for Lithuania: 101464403; European Union: 91000002
LINKEDIN_GEO_IDS: list[str] = ["101464403", "100565514", "101282230", "90010383"]  # LT, LV, EE, PL
LINKEDIN_KEYWORDS: list[str] = [
    "software engineer",
    "software developer",
    "data scientist",
    "machine learning engineer",
    "AI engineer",
    "backend developer",
    "full stack developer",
    "data engineer",
]
LINKEDIN_MAX_PAGES: int = int(os.getenv("LINKEDIN_MAX_PAGES", "10"))

# Temporal filter: only collect ads posted within this many days
MAX_POSTING_AGE_DAYS: int = int(os.getenv("MAX_POSTING_AGE_DAYS", "90"))

# ── Playwright timeout in milliseconds ────────────────────────────────────────
PAGE_TIMEOUT: int = 30_000
NAVIGATION_TIMEOUT: int = 60_000
