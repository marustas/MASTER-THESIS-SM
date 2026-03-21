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

# Playwright timeout in milliseconds
PAGE_TIMEOUT: int = 30_000
NAVIGATION_TIMEOUT: int = 60_000
