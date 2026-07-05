"""Verifier step: for cited outputs, check that the citation exists in the corpus
and that the passage actually supports the claim.

Two layers, and the ordering matters:

  1. Passage match — locate the citation in the corpus. A *deterministic*,
     auditable check that emits WHICH tier matched. Its central job is to verify
     the citation *exists*, not merely that it is well-formed: a well-shaped
     "§6.999" that is nowhere in the corpus is the classic hallucinated citation,
     and shape-validation alone sails straight past it.

  2. Topical alignment — a narrow LLM-as-judge call returning
     supports / unrelated / contradicts / uncertain. Only consulted once a passage
     is located; catches *miscited* (real-passage, wrong-topic) citations.

Design choices carried over from a production clinical pipeline:
  - Three-valued passage match: no corpus -> UNVERIFIABLE, never a silent pass.
  - **Location-aware**: when a citation names a section, the passage must appear *within
    that section's window*, not merely somewhere in the corpus. This catches the citation
    that quotes real text but attributes it to the wrong (or a non-existent) section — a
    class of miscitation a whole-corpus bag-of-words match is blind to.
  - The judge is skipped when the passage cannot be located (its verdict is moot),
    saving a call and avoiding a misleading "supports" on a fabricated citation.

Fuzzy matching uses `rapidfuzz` when installed (`pip install cite-or-abstain[fuzzy]`) and
falls back to the stdlib `difflib` otherwise. For a large embedded corpus, swap in a
vector-similarity match by replacing `locate_passage`.
"""
from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path

from .llm.base import LLMBackend
from .types import Citation, PassageMatch, TopicalAlignment, VerificationResult

try:  # optional acceleration/quality; stdlib difflib is the default fallback
    from rapidfuzz import fuzz as _rapidfuzz
except ImportError:  # pragma: no cover - exercised only when the extra is absent
    _rapidfuzz = None

# Hard cap on how much corpus after a section anchor counts as "in that section".
_SECTION_WINDOW = 1500

# Shape of a section header, used to find where the *next* section begins so the window
# for one section does not bleed into the next. Covers "§6.4.2", "6.4.2 " at line start,
# and markdown "## " headers. MULTILINE so ^ anchors match at each line start.
_NEXT_ANCHOR_RE = re.compile(
    r"(?:§\s*\d+(?:\.\d+)*)|(?:^\s{0,3}\d+(?:\.\d+)+\s)|(?:^#{1,6}\s)",
    re.MULTILINE,
)


def _section_window(corpus: str, idx: int, section_len: int) -> str:
    """The slice of corpus belonging to the section located at `idx`.

    Runs from the section anchor to the next section header (or the hard cap), so a
    passage from a *later* section is not counted as being in this one.
    """
    start = idx
    m = _NEXT_ANCHOR_RE.search(corpus, idx + section_len)
    end = m.start() if m else len(corpus)
    return corpus[start : min(end, start + _SECTION_WINDOW)]


def _partial_ratio(needle: str, haystack: str) -> float:
    """Best fuzzy match of `needle` anywhere inside `haystack`, in [0, 1].

    rapidfuzz if available; otherwise a stdlib difflib sliding-window approximation.
    """
    if not needle or not haystack:
        return 0.0
    if _rapidfuzz is not None:
        return _rapidfuzz.partial_ratio(needle, haystack) / 100.0
    n = len(needle)
    if n >= len(haystack):
        return difflib.SequenceMatcher(None, needle, haystack).ratio()
    best = 0.0
    step = max(1, n // 3)
    for i in range(0, len(haystack) - n + 1, step):
        r = difflib.SequenceMatcher(None, needle, haystack[i : i + n]).ratio()
        if r > best:
            best = r
            if best >= 0.995:
                break
    return best


def _passage_in_region(
    passage: str, region: str, fuzzy_threshold: float, scoped: bool = False
) -> tuple[bool, str]:
    """Is `passage` present in `region`? Returns (found, method).

    Substring -> fuzzy. When `scoped` (the region is a *single located section*, not the
    whole corpus), a token-overlap tier is added: within one section, high overlap of the
    passage's content words is a reliable signal that a paraphrase belongs there. This is
    NOT applied to whole-corpus search, where bag-of-words would be imprecise.
    """
    p = passage.strip().lower().strip("\"'")
    if not p:
        return (False, "")
    snippet = p[:200]
    region_lower = region.lower()
    if snippet in region_lower:
        return (True, "substring")
    if _partial_ratio(snippet, region_lower) >= fuzzy_threshold:
        return (True, "fuzzy")
    if scoped:
        tokens = set(re.findall(r"[a-z0-9]{4,}", p))
        if len(tokens) >= 4:
            region_tokens = set(re.findall(r"[a-z0-9]{4,}", region_lower))
            if len(tokens & region_tokens) / len(tokens) >= 0.7:
                return (True, "section-token")
    return (False, "")

PROMPT_VERSION = 1
_PROMPT_PATH = Path(__file__).parent / "prompts" / "verify_alignment.txt"
_VERIFIER_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
_PROMPT_SHA = hashlib.sha256(_VERIFIER_PROMPT.encode("utf-8")).hexdigest()[:12]


def prompt_fingerprint() -> dict[str, str]:
    """Pinned identity of the verifier prompt, for the run's frozen-judge record."""
    return {"verifier_prompt_version": str(PROMPT_VERSION), "verifier_prompt_sha": _PROMPT_SHA}


def locate_passage(
    citation: Citation, corpus: str | None, fuzzy_threshold: float = 0.80
) -> tuple[PassageMatch, str]:
    """Locate a citation in the corpus. Returns (match, method).

    Location-aware strategy:
      - No corpus -> UNVERIFIABLE (not the same as "not found").
      - Section cited but absent from the corpus -> NOT_FOUND (`section-absent`): the
        reference is fabricated regardless of whether similar prose appears elsewhere.
      - Section cited and present -> the passage must appear within that section's window
        (FOUND `section+substring/fuzzy`); if the section is real but the passage is not
        in it, that is a wrong-location citation -> NOT_FOUND (`section-mismatch`).
      - No section -> search the whole corpus for the passage (substring, then fuzzy).

    The winning tier is always reported so every verdict is auditable.
    """
    if not corpus:
        return (PassageMatch.UNVERIFIABLE, "no-corpus")

    corpus_lower = corpus.lower()

    if citation.section:
        section = citation.section.strip().lower()
        idx = corpus_lower.find(section)
        if idx == -1:
            # The cited section does not exist -> fabricated reference.
            return (PassageMatch.NOT_FOUND, "section-absent")
        if citation.passage:
            window = _section_window(corpus, idx, len(section))
            found, how = _passage_in_region(citation.passage, window, fuzzy_threshold, scoped=True)
            if found:
                return (PassageMatch.FOUND, f"section+{how}")
            # Section is real, but the passage is not in it: cited to the wrong place.
            return (PassageMatch.NOT_FOUND, "section-mismatch")
        # Section given, no passage: the section id existing is all we can confirm.
        return (PassageMatch.FOUND, "section-id")

    if citation.passage:
        found, how = _passage_in_region(citation.passage, corpus, fuzzy_threshold)
        if found:
            return (PassageMatch.FOUND, how)
        return (PassageMatch.NOT_FOUND, "none")

    # Nothing locatable (no section and no passage).
    return (PassageMatch.NOT_FOUND, "none")


def passage_in_corpus(citation: Citation, corpus: str, fuzzy_threshold: float = 0.80) -> bool:
    """Boolean convenience wrapper around `locate_passage` (True iff FOUND).

    Retained as the documented swap point; production users can replace this or
    `locate_passage` with rapidfuzz / pgvector without touching the scorer.
    """
    match, _ = locate_passage(citation, corpus, fuzzy_threshold)
    return match == PassageMatch.FOUND


def verify_alignment(claim: str, citation: Citation, backend: LLMBackend) -> TopicalAlignment:
    """Use an LLM judge to determine whether the cited passage supports the claim."""
    user_prompt = (
        f"CLAIM:\n{claim}\n\n"
        f"CITED PASSAGE:\n"
        f"  source: {citation.source}\n"
        f"  section: {citation.section}\n"
        f"  text: {citation.passage}\n\n"
        "Does the cited passage support the claim? Respond with exactly one word."
    )
    response = backend.complete(
        system=_VERIFIER_PROMPT,
        user=user_prompt,
        temperature=0.0,
        max_tokens=256,  # room for the model to answer; too small yields empty -> false uncertain
    )
    text = response.text.strip().lower()

    # Find the first alignment word that appears in the response.
    for ta in TopicalAlignment:
        if ta.value in text:
            return ta
    return TopicalAlignment.UNCERTAIN


def verify_citation(
    claim: str,
    citation: Citation,
    corpus: str | None,
    backend: LLMBackend,
    fuzzy_threshold: float = 0.80,
) -> VerificationResult:
    """Run the full verification pipeline on one citation: passage match + topical alignment.

    The alignment judge is only consulted when the passage is actually located;
    for an unlocatable (fabricated) or unverifiable citation the verdict is
    already determined by the passage match, so spending a judge call would only
    add a misleading signal.
    """
    match, method = locate_passage(citation, corpus, fuzzy_threshold)

    if match == PassageMatch.FOUND:
        alignment = verify_alignment(claim, citation, backend)
    else:
        # Verdict derives entirely from the passage match; do not consult the judge.
        alignment = TopicalAlignment.UNCERTAIN

    result = VerificationResult(
        citation=citation,
        passage_match=match,
        topical_alignment=alignment,
        match_method=method,
    )
    result.verifier_rationale = (
        f"passage={match.value} via {method}; "
        f"alignment={alignment.value}; verdict={result.verdict.value}"
    )
    return result
