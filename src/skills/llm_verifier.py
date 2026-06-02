"""Local-LLM verifier for extracted ESCO skill URIs.

Wraps a small instruct-tuned causal LM to ask, for each candidate skill,
"does this document genuinely teach or require that skill?".  Outputs a
filtered list of URIs the model confirms as on-topic.

Designed as a post-filter on top of the existing extractor: it never
adds new URIs, only rejects existing ones.  Recall ceiling is therefore
the extractor's recall; the verifier's job is to improve precision.

Default model: ``Qwen/Qwen2.5-1.5B-Instruct`` (~3 GB, MPS-friendly).
Tests inject a callable instead of the real model — no network access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from loguru import logger

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_BATCH_SIZE = 10
DEFAULT_DOC_CHAR_LIMIT = 3500
DEFAULT_MAX_NEW_TOKENS = 256


# A "model callable" signature: takes a prompt string, returns text response.
ModelCallable = Callable[[str], str]


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of verifying one (doc, skill) pair."""

    esco_uri: str
    preferred_label: str
    kept: bool
    raw_answer: str


def build_prompt(doc_text: str, labels: list[str], char_limit: int) -> str:
    """Format the verification prompt for a batch of candidate labels."""
    snippet = doc_text.strip()
    if len(snippet) > char_limit:
        snippet = snippet[:char_limit] + " […]"
    numbered = "\n".join(f"{i + 1}. {label}" for i, label in enumerate(labels))
    return (
        "You are reviewing extracted skills for a document. The document is "
        "either a study programme description or a job advertisement. For each "
        "candidate skill, decide whether it plausibly belongs to what the document "
        "teaches (programme) or requires (job).\n"
        "\n"
        "Document:\n"
        f"{snippet}\n"
        "\n"
        "Answer YES for a candidate if ANY of the following holds:\n"
        "  - the skill or a close synonym appears in the document, or\n"
        "  - the skill is a standard expected competence in the document's "
        "domain (e.g. programming for a Computer Science programme; cyber "
        "security for an Information Security job), or\n"
        "  - the skill is reasonably implied by the document's scope and "
        "learning outcomes.\n"
        "\n"
        "Answer NO only when the skill is clearly from a different domain or "
        "is otherwise irrelevant (e.g. agroforestry for a software programme; "
        "audiology for a programming job; tutor students in a job advertisement "
        "that is not in education).\n"
        "\n"
        "When in doubt, prefer YES.\n"
        "\n"
        "Candidate skills:\n"
        f"{numbered}\n"
        "\n"
        f"Respond with exactly {len(labels)} lines, one per candidate, in the format:\n"
        "1. YES\n"
        "2. NO\n"
        "...\n"
        "Output nothing else.\n"
    )


# Match a numbered line, then find the first YES/NO that appears on that line —
# skipping markdown decoration like **YES**, _NO_, "YES", etc.
_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[.):\-]\s*(.+)$", re.MULTILINE)
_VERDICT_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def parse_answers(response: str, n: int) -> list[bool]:
    """Parse the model response into a length-n list of bool decisions.

    For each numbered line in the response, scan for the first YES/NO
    token (ignoring markdown decoration). Missing or unparseable
    answers default to True (keep) so the verifier is a strict
    precision filter — only confident NOs are rejected.
    """
    answers: dict[int, bool] = {}
    for idx_str, tail in _NUMBERED_LINE_RE.findall(response or ""):
        idx = int(idx_str)
        if not (1 <= idx <= n):
            continue
        m = _VERDICT_RE.search(tail)
        if m:
            answers[idx] = m.group(1).upper() == "YES"
    return [answers.get(i + 1, True) for i in range(n)]


def verify_candidates(
    doc_text: str,
    candidates: list[tuple[str, str]],
    model_call: ModelCallable,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    char_limit: int = DEFAULT_DOC_CHAR_LIMIT,
) -> list[VerifyResult]:
    """Run the verifier over one document's candidate URIs.

    Parameters
    ----------
    doc_text:
        Cleaned text of the target document.
    candidates:
        List of ``(esco_uri, preferred_label)`` tuples to evaluate.
    model_call:
        Callable taking a prompt string and returning the model's text
        response.  Injected so tests can pass a deterministic mock.
    batch_size:
        Candidates per LLM call.

    Returns
    -------
    A list of ``VerifyResult`` in the same order as ``candidates``.
    """
    if not candidates:
        return []

    results: list[VerifyResult] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        labels = [lbl for _, lbl in batch]
        prompt = build_prompt(doc_text, labels, char_limit=char_limit)
        response = model_call(prompt)
        decisions = parse_answers(response, len(batch))
        for (uri, label), kept in zip(batch, decisions):
            results.append(VerifyResult(
                esco_uri=uri, preferred_label=label, kept=kept, raw_answer=response,
            ))
    return results


def keep_only(results: Iterable[VerifyResult]) -> list[tuple[str, str]]:
    """Convenience: return the (uri, label) tuples the verifier kept."""
    return [(r.esco_uri, r.preferred_label) for r in results if r.kept]


# ── Real-model loader (separate from the verification logic) ─────────────────


def load_transformers_model_call(
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> ModelCallable:
    """Load a HuggingFace causal LM and return a callable suitable for
    ``verify_candidates``.  Imported lazily so unit tests stay offline.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading {model_name} on {device}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device)
    model.eval()
    logger.info("Model ready.")

    def _call(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _call
