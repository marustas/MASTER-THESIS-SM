"""
Text cleaning utilities.

Handles:
  - HTML / Markdown stripping
  - Whitespace and unicode normalization
  - Boilerplate pattern removal (cookie banners, legal footers, nav fragments)
  - Minimum length filtering
"""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

# ── Boilerplate patterns ───────────────────────────────────────────────────────
# Regex patterns matched line-by-line; matching lines are dropped.
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r, re.IGNORECASE)
    for r in [
        r"^(all rights reserved|copyright\s*©)",
        r"^(cookie policy|privacy policy|terms of (use|service))",
        r"we use cookies",
        r"subscribe to our newsletter",
        r"follow us on (linkedin|twitter|facebook)",
        r"^\s*share\s+(this\s+)?(job|post|article)\s*$",
        r"^\s*(apply now|send cv|siųsti cv)\s*$",
        r"^\s*(back to (top|search|results))\s*$",
        r"lygios galimybės",            # Lithuanian equal-opportunity boilerplate
        r"mes garantuojame",
        r"^\s*[\|\-–—•]+\s*$",          # separator lines
        r"^\s*\d+\s*$",                 # bare numbers (pagination artefacts)
    ]
]

# Patterns whose presence anywhere in the line triggers removal
_INLINE_BOILERPLATE: list[re.Pattern] = [
    re.compile(r, re.IGNORECASE)
    for r in [
        r"click here to apply",
        r"equal opportunity employer",
        r"<\s*/?[a-z][^>]{0,100}>",     # residual HTML tags after BS4
    ]
]


def strip_html(text: str) -> str:
    """Remove all HTML/XML markup and decode HTML entities."""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ")


def normalize_unicode(text: str) -> str:
    """NFC-normalize unicode; replace non-breaking spaces and zero-width chars."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")   # non-breaking space
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\u200c", "")  # zero-width non-joiner
    text = text.replace("\ufeff", "")  # BOM
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs; normalize line endings; strip edges."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_boilerplate_lines(text: str) -> str:
    """Drop lines that match known boilerplate patterns."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if any(p.search(stripped) for p in _BOILERPLATE_PATTERNS):
            continue
        if any(p.search(stripped) for p in _INLINE_BOILERPLATE):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


_LAMA_BPO_MARKER = "Programmes granting same qualifications"


def strip_lama_bpo_nav(text: str) -> str:
    """Remove LAMA BPO navigation boilerplate that precedes programme content."""
    idx = text.find(_LAMA_BPO_MARKER)
    if idx < 0:
        return text
    return text[idx + len(_LAMA_BPO_MARKER):].lstrip()


# ── LinkedIn boilerplate ──────────────────────────────────────────────────────

# Section headers that mark the start of non-technical content.
# Everything from the *first* match of these to the end of the text is dropped.
_LINKEDIN_CUTOFF_RE = re.compile(
    r"^\s*("
    r"what we offer"
    r"|we offer"
    r"|our offer"
    r"|what you('ll| will) get"
    r"|benefits"
    r"|benefits for you"
    r"|perks"
    r"|equal opportunity"
    r"|salary\s*:.*"
    r"|compensation\s*:.*"
    r"|atlyginimas\s*:.*"
    r"|siųsdami savo gyvenimo aprašymą"
    r"|kandidatų konfidencialumą"
    r"|joinrs ai"
    r")\s*$",
    re.IGNORECASE,
)

# Inline patterns: if found anywhere in a line, that line triggers cutoff.
# These must be specific enough to avoid matching legitimate technical content.
_LINKEDIN_CUTOFF_INLINE_RE = re.compile(
    r"we are proud to (foster|be an equal)"
    r"|proud to be an equal opportunity"
    r"|we (provide|ensure) equal opportunity"
    r"|workplace free from discrimination"
    r"|committed to (fostering|building) an inclusive",
    re.IGNORECASE,
)

# Header line present in virtually all LinkedIn job posts.
_LINKEDIN_HEADER_RE = re.compile(r"^\s*about the job\s*$", re.IGNORECASE)


def strip_linkedin_boilerplate(text: str) -> str:
    """Remove LinkedIn-specific boilerplate from job descriptions.

    1. Strips the universal "About the job" header.
    2. Truncates everything from the first non-technical section
       (benefits, EEO, salary, data-protection) to the end.
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        if _LINKEDIN_HEADER_RE.match(line.strip()):
            continue
        if _LINKEDIN_CUTOFF_RE.match(line.strip()):
            break
        if _LINKEDIN_CUTOFF_INLINE_RE.search(line):
            break
        cleaned.append(line)
    return "\n".join(cleaned)


def remove_urls(text: str) -> str:
    """Strip bare URLs from text (keep surrounding context)."""
    return re.sub(r"https?://\S+", "", text)


def clean(
    text: str,
    *,
    strip_html_tags: bool = True,
    remove_urls_flag: bool = True,
    min_length: int = 50,
) -> str | None:
    """
    Full cleaning pass for a single text string.

    Returns the cleaned string, or None if the result is shorter than
    `min_length` characters (signals insufficient content for further processing).
    """
    if not text or not text.strip():
        return None

    if strip_html_tags:
        text = strip_html(text)

    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    text = remove_boilerplate_lines(text)

    if remove_urls_flag:
        text = remove_urls(text)

    text = normalize_whitespace(text)  # second pass after removals

    if len(text) < min_length:
        return None

    return text
