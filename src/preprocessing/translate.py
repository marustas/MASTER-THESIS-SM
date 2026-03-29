"""
Lithuanian → English translation utility.

Uses the Helsinki-NLP/opus-mt-lt-en model from HuggingFace Transformers.
The model (~300 MB) is downloaded on first use and cached by HuggingFace.

Usage:
    from src.preprocessing.translate import translate_lt_to_en
    en_text = translate_lt_to_en("Ieškome programuotojo...")
"""

from __future__ import annotations

from loguru import logger

# Maximum characters per chunk sent to the model.
# opus-mt-lt-en has a 512 token limit; ~400 chars is a safe character ceiling.
_CHUNK_SIZE = 400

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading Helsinki-NLP/opus-mt-lt-en translation model...")
        _pipeline = hf_pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-lt-en",
            device=-1,  # CPU
        )
        logger.info("Translation model loaded.")
    return _pipeline


def translate_lt_to_en(text: str) -> str:
    """
    Translate Lithuanian text to English.

    Long texts are split into ~400-character chunks at sentence boundaries
    (period + space) to stay within the model's token limit.  Chunks are
    translated individually and joined with a space.

    Returns the original text unchanged if it is empty or translation fails.
    """
    if not text or not text.strip():
        return text

    chunks = _split_into_chunks(text, _CHUNK_SIZE)
    pipe = _get_pipeline()

    translated_parts: list[str] = []
    for chunk in chunks:
        try:
            result = pipe(chunk, max_length=512)
            translated_parts.append(result[0]["translation_text"])
        except Exception as exc:
            logger.warning(f"Translation chunk failed: {exc} — keeping original")
            translated_parts.append(chunk)

    return " ".join(translated_parts)


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars, preferring sentence breaks."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break
        # Find the last sentence boundary within max_chars
        split_at = text.rfind(". ", 0, max_chars)
        if split_at == -1:
            # No sentence boundary — split at last space
            split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            # No space either — hard split
            split_at = max_chars
        else:
            split_at += 1  # include the period/space in the current chunk

        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    return [c for c in chunks if c]
