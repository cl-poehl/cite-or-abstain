"""Verifier step: for cited outputs, check that the citation exists in the corpus
and that the passage actually supports the claim.

Two layers:
  1. Passage match — fuzzy substring/token overlap against the corpus. Catches
     fully-hallucinated citations.
  2. Topical alignment — LLM-as-judge call returning supports / unrelated /
     contradicts / uncertain. Catches miscited (real-passage, wrong-topic) citations.

The passage matcher here is intentionally simple. For production use, swap in
rapidfuzz or a vector-similarity match against an embedded corpus.
"""
from __future__ import annotations

from pathlib import Path

from .llm.base import LLMBackend
from .types import Citation, TopicalAlignment, VerificationResult

_PROMPT_PATH = Path(__file__).parent / "prompts" / "verify_alignment.txt"
_VERIFIER_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def passage_in_corpus(citation: Citation, corpus: str, token_threshold: float = 0.6) -> bool:
    """Return True if the citation's passage or section is locatable in the corpus.

    Match strategy, in order:
      1. Direct substring of the first ~80 characters of the passage.
      2. Section identifier substring match.
      3. Token-overlap fallback for paraphrased passages.

    This is intentionally simple. For production grade, see rapidfuzz / pgvector.
    """
    if not corpus:
        return False

    corpus_lower = corpus.lower()

    # 1. Direct substring of opening of passage.
    if citation.passage:
        snippet = citation.passage[:80].lower().strip().strip("\"'")
        if snippet and snippet in corpus_lower:
            return True

    # 2. Section identifier match (e.g., "§6.4.2").
    if citation.section:
        section = citation.section.lower().strip()
        if section and section in corpus_lower:
            return True

    # 3. Token-overlap fallback.
    if citation.passage:
        passage_tokens = {
            t for t in citation.passage.lower().split() if len(t) > 3
        }
        if len(passage_tokens) >= 3:
            corpus_tokens = set(corpus_lower.split())
            overlap = passage_tokens & corpus_tokens
            return (len(overlap) / len(passage_tokens)) >= token_threshold

    return False


def verify_alignment(
    claim: str, citation: Citation, backend: LLMBackend
) -> TopicalAlignment:
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
        max_tokens=20,
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
) -> VerificationResult:
    """Run the full verification pipeline on one citation: passage match + topical alignment."""
    passage_found = passage_in_corpus(citation, corpus) if corpus else True
    alignment = verify_alignment(claim, citation, backend)
    return VerificationResult(
        citation=citation,
        passage_found=passage_found,
        topical_alignment=alignment,
        verifier_rationale=(
            f"passage_match={'yes' if passage_found else 'no'}, "
            f"alignment={alignment.value}"
        ),
    )
