"""
Step 5 — Semantic Embedding Generation.

Produces transformer-based dense embeddings for all preprocessed study
programme descriptions and job advertisements.  The embeddings are stored
as a new column in the same parquet files produced by Step 3.

For programmes two embedding columns are generated:
  embedding_brief    — embedding of brief_description (LAMA BPO registry text)
  embedding_extended — embedding of extended_description (university website text)
  embedding          — embedding of cleaned_text (combined, used for alignment)

For job ads a single column is generated:
  embedding          — embedding of cleaned_text

All embeddings are L2-normalised (unit vectors) so downstream cosine
similarity is a simple dot product.

Model: all-MiniLM-L6-v2 (default)  — 384 dimensions, fast, strong on short texts.
The model name is configurable so a larger model can be swapped in for
experiments without changing any downstream code.

Usage:
    python -m src.embeddings.generator
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.scraping.config import DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_PROGRAMMES = (
    DATA_DIR / "processed" / "programmes" / "programmes_preprocessed.parquet"
)
PROCESSED_JOBS = (
    DATA_DIR / "processed" / "job_ads" / "jobs_preprocessed.parquet"
)
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

PROGRAMMES_OUT = EMBEDDINGS_DIR / "programmes_embeddings.parquet"
JOBS_OUT = EMBEDDINGS_DIR / "jobs_embeddings.parquet"

DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


# ── Core embedding helper ──────────────────────────────────────────────────────

def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Encode a list of texts to L2-normalised embeddings.

    Empty / None texts are replaced with a zero vector of the same dimension;
    downstream code treats zero-norm embeddings as "no text available".
    """
    # Identify non-empty positions
    placeholder = "__EMPTY__"
    safe_texts = [t if (t and t.strip()) else placeholder for t in texts]
    non_empty_mask = [t != placeholder for t in safe_texts]

    # Encode only non-empty texts for efficiency
    non_empty_texts = [t for t, ok in zip(safe_texts, non_empty_mask) if ok]

    if non_empty_texts:
        encoded = model.encode(
            non_empty_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        dim = encoded.shape[1]
    else:
        dim = model.get_sentence_embedding_dimension()
        encoded = np.zeros((0, dim), dtype=np.float32)

    result = np.zeros((len(texts), dim), dtype=np.float32)
    non_empty_idx = 0
    for i, ok in enumerate(non_empty_mask):
        if ok:
            result[i] = encoded[non_empty_idx]
            non_empty_idx += 1
    return result


def _embeddings_to_list(arr: np.ndarray) -> list[list[float]]:
    """Convert numpy array rows to Python lists for parquet storage."""
    return arr.tolist()


# ── Programme embeddings ───────────────────────────────────────────────────────

def embed_programmes(
    model: SentenceTransformer,
    input_path: Path = PROCESSED_PROGRAMMES,
    output_path: Path = PROGRAMMES_OUT,
) -> pd.DataFrame:
    """
    Add embedding columns to the preprocessed programmes parquet.

    Columns added:
      embedding          — from cleaned_text (combined description)
      embedding_brief    — from brief_description
      embedding_extended — from extended_description
    """
    logger.info(f"Loading programmes from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"  {len(df)} programme records")

    logger.info("Generating combined embeddings (cleaned_text) …")
    df["embedding"] = _embeddings_to_list(
        embed_texts(model, df["cleaned_text"].fillna("").tolist())
    )

    if "brief_description" in df.columns:
        logger.info("Generating brief-description embeddings …")
        df["embedding_brief"] = _embeddings_to_list(
            embed_texts(model, df["brief_description"].fillna("").tolist())
        )

    if "extended_description" in df.columns:
        logger.info("Generating extended-description embeddings …")
        df["embedding_extended"] = _embeddings_to_list(
            embed_texts(model, df["extended_description"].fillna("").tolist())
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved → {output_path}  ({len(df)} rows)")
    return df


# ── Job ad embeddings ──────────────────────────────────────────────────────────

def embed_job_ads(
    model: SentenceTransformer,
    input_path: Path = PROCESSED_JOBS,
    output_path: Path = JOBS_OUT,
) -> pd.DataFrame:
    """
    Add an embedding column to the preprocessed job ads parquet.

    Column added:
      embedding — from cleaned_text
    """
    logger.info(f"Loading job ads from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"  {len(df)} job ad records")

    logger.info("Generating job ad embeddings (cleaned_text) …")
    df["embedding"] = _embeddings_to_list(
        embed_texts(model, df["cleaned_text"].fillna("").tolist())
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved → {output_path}  ({len(df)} rows)")
    return df


# ── CLI entry-point ────────────────────────────────────────────────────────────

def run(model_name: str = DEFAULT_MODEL) -> None:
    logger.info(f"Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    embed_programmes(model)
    embed_job_ads(model)

    logger.info("Step 5 complete — embeddings saved to data/embeddings/")


if __name__ == "__main__":
    run()
