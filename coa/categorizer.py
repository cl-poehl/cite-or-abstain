"""Categorizer step: assign an LLM output to one of the four cite-or-abstain categories.

Uses an LLM-as-judge call against a versioned prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

from .llm.base import LLMBackend
from .types import Categorization, Category, Citation

_PROMPT_PATH = Path(__file__).parent / "prompts" / "categorize.txt"
_CATEGORIZER_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def _strip_code_fences(text: str) -> str:
    """LLMs love to wrap JSON in ```json fences even when told not to. Strip them."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence and optional language tag
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        # Drop the closing fence
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    return text


def categorize(output: str, backend: LLMBackend) -> Categorization:
    """Categorize an LLM output as cited / uncited-confident / uncited-hedged / abstained.

    Returns a Categorization with citations extracted (empty unless cited) and a rationale.
    If the categorizer's own response fails to parse, falls back to uncited-hedged with the
    raw text in the rationale.
    """
    user_prompt = (
        "OUTPUT TO CATEGORIZE:\n\n"
        f"{output}\n\n"
        "Respond with JSON only."
    )
    response = backend.complete(
        system=_CATEGORIZER_PROMPT,
        user=user_prompt,
        temperature=0.0,
        max_tokens=800,
    )

    try:
        text = _strip_code_fences(response.text)
        data = json.loads(text)
        category = Category(data["category"])
        citations = [Citation(**c) for c in data.get("citations", [])]
        rationale = data.get("rationale", "")
        # Defensive: enforce empty citations for non-cited categories.
        if category != Category.CITED:
            citations = []
        return Categorization(category=category, citations=citations, rationale=rationale)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return Categorization(
            category=Category.UNCITED_HEDGED,
            citations=[],
            rationale=f"[categorizer-parse-error: {e!s}] raw={response.text[:240]!r}",
        )
