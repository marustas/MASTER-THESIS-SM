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


# ── Section-weighted programme embedding ──────────────────────────────────────

# Section groups and their weights for programme embeddings.
# Most discriminative sections get the highest weight.
# _remainder catches content under unmapped headers (meta sections like
# "activities of teaching and learning", career paths, assessment methods).
SECTION_WEIGHTS = {
    "subjects": 0.35,         # course lists — most discriminative
    "outcomes": 0.25,         # learning outcomes / competencies
    "identity": 0.15,         # objectives + distinctive features
    "specialisations": 0.20,  # domain-specific tracks — key differentiator
    "_remainder": 0.05,       # unmapped content — small weight so it's not lost
}

# Map raw section headers to groups.
# Headers are normalised to lower-case with trailing colon stripped.
_SECTION_MAP: dict[str, str] = {
    # ── subjects ──────────────────────────────────────────────────────────
    "study subjects (modules), practical training": "subjects",
    "study subjects (modules)": "subjects",
    "study subjects, practical training": "subjects",
    "study course units, practical training": "subjects",
    "study modules (subjects), practical training": "subjects",
    "subjects": "subjects",
    "framework": "subjects",
    "framework: course units, practice": "subjects",
    "framework: study subjects, practical training": "subjects",
    "the study field course units": "subjects",
    "students study the following subjects (modules)": "subjects",
    "i. general subjects of collegiate studies": "subjects",
    "i. general study subjects": "subjects",
    "general college study subjects (15 ects)": "subjects",
    "subjects in the study field (99 ects)": "subjects",
    "compulsory subjects": "subjects",
    "main study subjects (165 credits)": "subjects",
    "arbitrary study subjects (60 credits)": "subjects",
    "study field subjects – 152 ects credits": "subjects",
    "general and/or specific study subjects – 19 ects credits": "subjects",
    "fundamental part of the study program (60 ects credits)": "subjects",
    "scope of study field subjects – 123 credits, of them": "subjects",
    "scope of study field subjects – 83 credits": "subjects",
    "scope of study field subjects – 125 credits, of them": "subjects",
    "scope of study field subjects – 80 credits": "subjects",
    "mandatory subjects – 10 credits": "subjects",
    "professional activity internships – 30 ects credits": "subjects",
    "practice (36 credits)": "subjects",
    "practice - 30 ects credits": "subjects",
    "iii. practices / internships": "subjects",
    "iv. final project": "subjects",
    # ── outcomes ──────────────────────────────────────────────────────────
    "learning outcomes": "outcomes",
    "study outcomes": "outcomes",
    "knowledge and its application": "outcomes",
    "research skills": "outcomes",
    "specific skills": "outcomes",
    "special skills": "outcomes",
    "social skills": "outcomes",
    "personal skills": "outcomes",
    "special abilities": "outcomes",
    "social abilities": "outcomes",
    "personal abilities": "outcomes",
    "ability to exploration": "outcomes",
    "graduates will be able to": "outcomes",
    "the graduates of this study programme will be able": "outcomes",
    "students completing this program will be able to": "outcomes",
    "after graduating from the studies, a person will be able": "outcomes",
    "after completion of the study programme, a graduate": "outcomes",
    "students will learn": "outcomes",
    "students will develop research skills and will be able to": "outcomes",
    "students will develop special skills": "outcomes",
    "students will develop such social skills": "outcomes",
    "students will develop following personal skills": "outcomes",
    # ── identity ──────────────────────────────────────────────────────────
    "objective(s) of a study programme": "identity",
    "objective of a study programme": "identity",
    "objectives of the study programme": "identity",
    "aim(s) of the study programme": "identity",
    "study programme abstract": "identity",
    "description of the study programme": "identity",
    "general description": "identity",
    "distinctive features of a study programme": "identity",
    "distinctive features of the study programme": "identity",
    "a) the programme focuses on": "identity",
    "b) unique content of the programme": "identity",
    "c) experienced teachers": "identity",
    # ── specialisations ───────────────────────────────────────────────────
    "specialisations": "specialisations",
    "specializations": "specialisations",
    "optional courses": "specialisations",
    "you can optionally choose up to 66 credits": "specialisations",
    "students can choose one out of two specializations from the study programme": "specialisations",
    "deeper specialization in the software systems field - 60 credits, of them": "specialisations",
    # Named specialisation modules → specialisations
    "management of computer systems specialization": "specialisations",
    "management of internet projects specialization": "specialisations",
    "programming for mobile devices specialization": "specialisations",
    "management of computer systems (module)": "specialisations",
    "management of internet projects (module)": "specialisations",
    "programming for mobile devices (module)": "specialisations",
    "1. specialisation. \u201ecyberphysical systems\u201c - 23 credits": "specialisations",
    "2. specialisation. \u201ecyber security\u201c - 23 credits": "specialisations",
    "free elective subjects - 9 credits": "specialisations",
    "specialization – machine learning": "specialisations",
    "specialization – web technologies": "specialisations",
    "specialization – business systems technology": "specialisations",
    # ── deliberately unmapped (non-content / meta) ────────────────────────
    # These go to _remainder intentionally:
    #   "activities of teaching and learning" (40)
    #   "access to further study" (40)
    #   "access to professional activity" (39)
    #   "methods of assessment of learning achievements" (37)
    #   "access to professional activity or further study" (32)
    #   "graduates work as" (1)
    #   … and other meta / career-path headers
}


def parse_programme_sections(text: str) -> dict[str, str]:
    """
    Parse a programme cleaned_text into section groups.

    Returns {group_name: concatenated_text} for each group in SECTION_WEIGHTS.
    Sections not matching any known header go into a '_remainder' key.
    """
    groups: dict[str, list[str]] = {g: [] for g in SECTION_WEIGHTS}

    current_group = "_remainder"
    for line in text.split("\n"):
        stripped = line.strip()
        # Detect section header: ends with colon, <80 chars
        if stripped.endswith(":") and 3 < len(stripped) < 80:
            header_key = stripped[:-1].strip().lower()
            # Check specialisation-prefixed headers like "Specialization - InfoSec"
            if header_key.startswith("specializ"):
                current_group = "specialisations"
            else:
                current_group = _SECTION_MAP.get(header_key, "_remainder")
        else:
            groups[current_group].append(line)

    return {g: "\n".join(lines).strip() for g, lines in groups.items()}


def embed_programme_sections(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Produce a weighted-section embedding for each programme text.

    Each text is split into section groups, each group is embedded independently
    (avoiding truncation), and the final embedding is the weighted average
    (L2-normalised).
    """
    dim = model.get_sentence_embedding_dimension()
    n = len(texts)
    result = np.zeros((n, dim), dtype=np.float32)

    # Parse all texts into sections
    all_sections = [parse_programme_sections(t) for t in texts]

    # Identify programmes where no named sections were found — fall back to
    # embedding the full text so they don't get a zero vector.
    named_groups = [g for g in SECTION_WEIGHTS if g != "_remainder"]
    no_sections = [
        i for i, s in enumerate(all_sections)
        if all(not s.get(g, "").strip() for g in named_groups)
    ]

    for group, weight in SECTION_WEIGHTS.items():
        group_texts = [s.get(group, "") for s in all_sections]
        group_embs = embed_texts(model, group_texts, batch_size=batch_size)
        result += weight * group_embs

    # Fall back: programmes without sections get a plain full-text embedding
    if no_sections:
        fallback_texts = [texts[i] for i in no_sections]
        fallback_embs = embed_texts(model, fallback_texts, batch_size=batch_size)
        for j, i in enumerate(no_sections):
            result[i] = fallback_embs[j]
        logger.info(f"  {len(no_sections)} programmes without sections — used full-text fallback")

    # L2-normalise the weighted average
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    result = result / norms

    return result


# ── Chunk-and-pool job embedding ──────────────────────────────────────────────

def embed_chunked(
    model: SentenceTransformer,
    texts: list[str],
    max_tokens: int = 256,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Embed long texts by splitting into chunks and mean-pooling.

    Each text is split into non-overlapping token-boundary chunks of
    ``max_tokens``. Each chunk is embedded, and the final embedding is
    the mean of all chunk embeddings (L2-normalised).
    """
    dim = model.get_sentence_embedding_dimension()
    tokenizer = getattr(model, "tokenizer", None)

    # Fall back to plain embedding if no tokenizer (e.g. mock models in tests)
    if tokenizer is None:
        return embed_texts(model, texts, batch_size=batch_size)

    # Tokenize all texts to find chunk boundaries (in characters)
    all_chunks: list[list[str]] = []
    for text in texts:
        if not text or not text.strip():
            all_chunks.append([])
            continue
        # Encode to token IDs, then decode in chunks
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for start in range(0, len(token_ids), max_tokens):
            chunk_ids = token_ids[start : start + max_tokens]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text)
        all_chunks.append(chunks if chunks else [])

    # Flatten all chunks for batch encoding
    flat_chunks = []
    chunk_map: list[tuple[int, int]] = []  # (doc_idx, chunk_idx_in_flat)
    for doc_idx, chunks in enumerate(all_chunks):
        for chunk in chunks:
            chunk_map.append((doc_idx, len(flat_chunks)))
            flat_chunks.append(chunk)

    if flat_chunks:
        flat_embs = model.encode(
            flat_chunks,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    else:
        flat_embs = np.zeros((0, dim), dtype=np.float32)

    # Mean-pool per document
    result = np.zeros((len(texts), dim), dtype=np.float32)
    doc_counts = np.zeros(len(texts), dtype=np.int32)
    for doc_idx, flat_idx in chunk_map:
        result[doc_idx] += flat_embs[flat_idx]
        doc_counts[doc_idx] += 1

    # Normalise
    for i in range(len(texts)):
        if doc_counts[i] > 0:
            result[i] /= doc_counts[i]
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    result = result / norms

    return result


# ── Programme embeddings ───────────────────────────────────────────────────────

def embed_programmes(
    model: SentenceTransformer,
    input_path: Path = PROCESSED_PROGRAMMES,
    output_path: Path = PROGRAMMES_OUT,
    use_sections: bool = True,
) -> pd.DataFrame:
    """
    Add embedding columns to the preprocessed programmes parquet.

    Columns added:
      embedding          — section-weighted (default) or from cleaned_text
      embedding_brief    — from brief_description
      embedding_extended — from extended_description
    """
    logger.info(f"Loading programmes from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"  {len(df)} programme records")

    cleaned = df["cleaned_text"].fillna("").tolist()

    if use_sections:
        logger.info("Generating section-weighted embeddings …")
        df["embedding"] = _embeddings_to_list(
            embed_programme_sections(model, cleaned)
        )
    else:
        logger.info("Generating combined embeddings (cleaned_text) …")
        df["embedding"] = _embeddings_to_list(
            embed_texts(model, cleaned)
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
    use_chunked: bool = True,
) -> pd.DataFrame:
    """
    Add an embedding column to the preprocessed job ads parquet.

    Column added:
      embedding — chunk-and-pool (default) or from cleaned_text
    """
    logger.info(f"Loading job ads from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"  {len(df)} job ad records")

    cleaned = df["cleaned_text"].fillna("").tolist()

    if use_chunked:
        logger.info("Generating chunked embeddings (chunk-and-pool) …")
        df["embedding"] = _embeddings_to_list(
            embed_chunked(model, cleaned)
        )
    else:
        logger.info("Generating job ad embeddings (cleaned_text) …")
        df["embedding"] = _embeddings_to_list(
            embed_texts(model, cleaned)
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
