"""Categorizer step: assign an LLM output to one of the four cite-or-abstain categories.

Uses an LLM-as-judge call against a versioned prompt. Two robustness lessons from
running LLM judges at scale are baked in:

  - Parsing is defensive. Judges wrap JSON in ``` fences, prepend <think>…</think>
    reasoning traces, and emit trailing prose. `_extract_json` strips those and
    scans for the first balanced-brace object rather than trusting `json.loads`
    on the raw text.

  - A parse failure is a *recorded outcome*, not a silent mislabel. v0.1 fell back
    to `uncited-hedged` on a parse error, which quietly polluted the scorecard with
    a real category. Now the categorization is flagged `parse_ok=False` and the
    scorer records the case as `judge-failed` (a harness-reliability signal),
    keeping it out of the model's coverage/failure rates.

Optionally runs the judge k times and takes a majority vote (a "frozen judge"
stabiliser). Default k=1: multi-draw self-consistency is *not* assumed to help a
categorical judgment — measure it before turning it on.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .llm.base import LLMBackend
from .types import Categorization, Category, Citation

PROMPT_VERSION = 1
_PROMPT_PATH = Path(__file__).parent / "prompts" / "categorize.txt"
_CATEGORIZER_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
_PROMPT_SHA = hashlib.sha256(_CATEGORIZER_PROMPT.encode("utf-8")).hexdigest()[:12]


def prompt_fingerprint() -> dict[str, str]:
    """Pinned identity of the categorizer prompt, for the run's frozen-judge record."""
    return {
        "categorizer_prompt_version": str(PROMPT_VERSION),
        "categorizer_prompt_sha": _PROMPT_SHA,
    }


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> reasoning traces that reasoning models prepend."""
    while "<think>" in text and "</think>" in text:
        start = text.index("<think>")
        end = text.index("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text


def _extract_json(text: str) -> str | None:
    """Best-effort extraction of the first complete JSON object from an LLM response.

    Strategy: strip think-tags and code fences, then scan for the first
    brace-balanced object (string/escape aware). Returns None if none is found.
    """
    text = _strip_think_tags(text).strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _categorize_once(output: str, backend: LLMBackend) -> Categorization:
    """Single categorizer call + defensive parse. parse_ok=False on any failure."""
    user_prompt = "OUTPUT TO CATEGORIZE:\n\n" f"{output}\n\n" "Respond with JSON only."
    response = backend.complete(
        system=_CATEGORIZER_PROMPT,
        user=user_prompt,
        temperature=0.0,
        max_tokens=800,
    )

    raw = _extract_json(response.text)
    if raw is None:
        return Categorization(
            category=Category.UNCITED_HEDGED,
            citations=[],
            rationale=f"[categorizer-parse-error: no JSON object] raw={response.text[:240]!r}",
            parse_ok=False,
        )
    try:
        data = json.loads(raw)
        category = Category(data["category"])
        citations = [Citation(**c) for c in data.get("citations", [])]
        rationale = data.get("rationale", "")
        # Defensive: enforce empty citations for non-cited categories.
        if category != Category.CITED:
            citations = []
        return Categorization(category=category, citations=citations, rationale=rationale)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        return Categorization(
            category=Category.UNCITED_HEDGED,
            citations=[],
            rationale=f"[categorizer-parse-error: {e!s}] raw={response.text[:240]!r}",
            parse_ok=False,
        )


def categorize(output: str, backend: LLMBackend, k: int = 1) -> Categorization:
    """Categorize an LLM output as cited / uncited-confident / uncited-hedged / abstained.

    Args:
        output: the LLM output to categorize.
        backend: the judge backend.
        k: number of judge draws. k=1 (default) is a single call. k>1 takes a
           majority vote over the category (a frozen-judge stabiliser); ties break
           toward the more conservative category. Draws that fail to parse do not
           vote; if *every* draw fails, the result is flagged parse_ok=False.

    Returns a Categorization; citations are populated only for `cited`.
    """
    if k <= 1:
        return _categorize_once(output, backend)

    draws = [_categorize_once(output, backend) for _ in range(k)]
    good = [d for d in draws if d.parse_ok]
    if not good:
        return draws[0]  # all failed -> propagate the parse failure

    # Majority vote on category; tie-break toward the most conservative (most dangerous) class.
    order = [
        Category.UNCITED_CONFIDENT,
        Category.UNCITED_HEDGED,
        Category.CITED,
        Category.ABSTAINED,
    ]
    tally = Counter(d.category for d in good)
    top = max(tally.values())
    winners = [c for c in order if tally.get(c, 0) == top]
    winning_category = winners[0]

    # Return the first good draw matching the winning category (keeps its citations/rationale).
    for d in good:
        if d.category == winning_category:
            return d
    return good[0]
