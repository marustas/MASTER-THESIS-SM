"""
Lithuanian → English translation utility.

Uses deep-translator (Google Translate backend) — no API key required,
uses the requests library (not httpx), so no conflict with Playwright.

Usage:
    from src.preprocessing.translate import translate_lt_to_en
    en_text = translate_lt_to_en("Ieškome programuotojo...")
"""

from __future__ import annotations

from loguru import logger

# Google Translate has a ~5000 character limit per request.
# Use a conservative ceiling to avoid hitting it with long descriptions.
_CHUNK_SIZE = 4500


def translate_lt_to_en(text: str) -> str:
    """
    Translate Lithuanian text to English via Google Translate.

    Long texts are split into ~4500-character chunks at sentence boundaries
    and translated individually.  Returns the original text unchanged if it
    is empty or every chunk fails.
    """
    if not text or not text.strip():
        return text

    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="lt", target="en")
    chunks = _split_into_chunks(text, _CHUNK_SIZE)
    translated_parts: list[str] = []

    for chunk in chunks:
        try:
            translated_parts.append(translator.translate(chunk))
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
        split_at = text.rfind(". ", 0, max_chars)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        else:
            split_at += 1

        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    return [c for c in chunks if c]
