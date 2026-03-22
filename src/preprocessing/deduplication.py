"""
Duplicate detection for programmes and job advertisements.

Two levels of deduplication:
  1. Exact — identical normalised text or identical URL/key.
  2. Near-duplicate — MinHash-based Jaccard similarity over character shingles.
     Records with similarity ≥ `NEAR_DUPLICATE_THRESHOLD` are considered
     duplicates; only the first occurrence is retained.

Near-duplicate detection is useful for job ads that are posted on multiple
platforms or re-posted with minor edits (different dates, slight wording changes).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

NEAR_DUPLICATE_THRESHOLD: float = 0.85
SHINGLE_SIZE: int = 3           # character n-gram size
NUM_HASH_FUNCTIONS: int = 128   # MinHash signature length


# ── Fingerprinting ─────────────────────────────────────────────────────────────

def text_fingerprint(text: str) -> str:
    """MD5 fingerprint of normalised (lowercased, whitespace-collapsed) text."""
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


def shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    """Return the set of k-character shingles of `text`."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    return {text[i : i + k] for i in range(max(0, len(text) - k + 1))}


# ── MinHash signature ──────────────────────────────────────────────────────────

def _hash_shingle(shingle: str, seed: int) -> int:
    """Deterministic hash of a shingle with a given seed."""
    return int(hashlib.md5(f"{seed}{shingle}".encode()).hexdigest(), 16)


def minhash_signature(text: str, num_hashes: int = NUM_HASH_FUNCTIONS) -> list[int]:
    """Compute a MinHash signature of length `num_hashes` for `text`."""
    shing = shingles(text)
    if not shing:
        return [0] * num_hashes
    sig = []
    for seed in range(num_hashes):
        min_hash = min(_hash_shingle(s, seed) for s in shing)
        sig.append(min_hash)
    return sig


def jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(a == b for a, b in zip(sig_a, sig_b))
    return matches / len(sig_a)


# ── Deduplicator ───────────────────────────────────────────────────────────────

@dataclass
class DeduplicationResult:
    kept: list[Any] = field(default_factory=list)
    removed_exact: int = 0
    removed_near: int = 0

    @property
    def total_removed(self) -> int:
        return self.removed_exact + self.removed_near

    def summary(self) -> str:
        return (
            f"Kept {len(self.kept)} records | "
            f"Removed {self.removed_exact} exact + {self.removed_near} near-duplicates"
        )


def deduplicate(
    records: list[Any],
    *,
    text_field: str,
    key_field: str | None = None,
    near_duplicate: bool = True,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> DeduplicationResult:
    """
    Deduplicate a list of dicts or Pydantic models.

    Parameters
    ----------
    records:
        List of dicts or Pydantic models.
    text_field:
        Attribute/key holding the main text to fingerprint.
    key_field:
        Optional attribute/key for an exact-match key (e.g. URL).
        Exact-key dedup runs before text fingerprinting.
    near_duplicate:
        Whether to run MinHash near-duplicate detection (slower).
    threshold:
        Jaccard similarity threshold for near-duplicates.
    """
    result = DeduplicationResult()
    seen_keys: set[str] = set()
    seen_fingerprints: set[str] = set()
    signatures: list[tuple[list[int], Any]] = []

    def _get(record: Any, attr: str) -> Any:
        if isinstance(record, dict):
            return record.get(attr)
        return getattr(record, attr, None)

    for record in records:
        # 1. Exact key dedup (URL or similar unique identifier)
        if key_field:
            key = _get(record, key_field)
            if key:
                key_norm = str(key).lower().split("?")[0]
                if key_norm in seen_keys:
                    result.removed_exact += 1
                    continue
                seen_keys.add(key_norm)

        # 2. Exact text fingerprint dedup
        text = _get(record, text_field) or ""
        if text:
            fp = text_fingerprint(text)
            if fp in seen_fingerprints:
                result.removed_exact += 1
                continue
            seen_fingerprints.add(fp)

        # 3. Near-duplicate detection via MinHash
        if near_duplicate and text:
            sig = minhash_signature(text)
            is_near_dup = False
            for existing_sig, _ in signatures:
                if jaccard_estimate(sig, existing_sig) >= threshold:
                    is_near_dup = True
                    break
            if is_near_dup:
                result.removed_near += 1
                continue
            signatures.append((sig, record))

        result.kept.append(record)

    logger.info(result.summary())
    return result
