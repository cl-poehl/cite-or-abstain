"""Optional semantic (embedding-based) passage matcher.

The default `coa.verifier.locate_passage` is **lexical** (substring → fuzzy). When a model
*paraphrases* a source — restating a guideline in its own words rather than quoting it —
lexical similarity can be low even though the claim is fully grounded, so a grounded
paraphrase is wrongly scored `fabricated`. (The bundled synthetic set includes paraphrase
cases that locate under this semantic matcher but not the lexical one.) A semantic matcher
compares *embeddings* instead and recovers those cases.

`SemanticMatcher(embed_fn)` is **signature-compatible with `locate_passage`**, so it is the
documented swap point: pass it as `locate=` to `verify_citation` / `score_case` /
`score_cases`. It deliberately preserves the *existence-verifying* semantics that make the
harness meaningful:

  - no corpus            -> UNVERIFIABLE  (never a silent pass)
  - cited section absent -> NOT_FOUND (`section-absent`): a fabricated reference is NOT
    resurrected by semantic similarity to prose elsewhere in the corpus
  - cited section present -> passage embedded against *that section's window* only
  - no section           -> passage embedded against the whole corpus

`embed_fn` is any callable ``list[str] -> list[list[float]]`` (a batch embedder): a local
sentence-transformers model (e.g. BGE-m3, fully on-prem) or an OpenAI-compatible
`/v1/embeddings` endpoint. coa ships **no** embedder, to stay dependency-light and
offline-capable — you inject the one your deployment already runs.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from .types import Citation, PassageMatch
from .verifier import _find_section_idx, _section_locator, _section_window

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    """Overlapping character windows over `text` (paragraph-agnostic, so it works on any
    corpus format). Overlap keeps a passage from being split across a chunk boundary."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


class SemanticMatcher:
    """Embedding-based passage matcher; a drop-in for `locate_passage`.

    Args:
        embed_fn: batch embedder, ``list[str] -> list[vector]``.
        threshold: cosine similarity at/above which a passage is considered located.
            Tune per embedding model; 0.60–0.70 is typical for multilingual sentence
            encoders. Report it — it is part of the verification method, like the fuzzy
            threshold is for the lexical matcher.
        chunk_chars / chunk_overlap: corpus windowing for whole-corpus search.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        threshold: float = 0.65,
        chunk_chars: int = 500,
        chunk_overlap: int = 120,
    ) -> None:
        self._embed = embed_fn
        self.threshold = threshold
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        # Cache corpus-chunk embeddings across citations in a run (corpus is fixed).
        self._corpus_cache: dict[int, tuple[list[str], list[Sequence[float]]]] = {}

    def _embedded_chunks(self, text: str) -> tuple[list[str], list[Sequence[float]]]:
        key = id(text)
        cached = self._corpus_cache.get(key)
        if cached is not None:
            return cached
        chunks = _chunk(text, self.chunk_chars, self.chunk_overlap)
        vecs = list(self._embed(chunks)) if chunks else []
        self._corpus_cache[key] = (chunks, vecs)
        return chunks, vecs

    def _best_sim(self, passage: str, region_chunks: list[str], region_vecs: Sequence) -> float:
        if not passage.strip() or not region_chunks:
            return 0.0
        pvec = list(self._embed([passage]))[0]
        return max((_cosine(pvec, cv) for cv in region_vecs), default=0.0)

    def __call__(
        self, citation: Citation, corpus: str | None, fuzzy_threshold: float = 0.80
    ) -> tuple[PassageMatch, str]:
        """Locate `citation` in `corpus` by embedding similarity. Returns (match, method).

        `fuzzy_threshold` is accepted for signature compatibility with `locate_passage`
        and ignored (semantic uses `self.threshold`).
        """
        if not corpus:
            return (PassageMatch.UNVERIFIABLE, "no-corpus")

        corpus_lower = corpus.lower()
        locator = _section_locator(citation.section)

        if locator:
            idx = _find_section_idx(corpus_lower, locator)
            if idx == -1:
                return (PassageMatch.NOT_FOUND, "section-absent")
            if citation.passage:
                window = _section_window(corpus, idx, len(locator))
                wchunks = _chunk(window, self.chunk_chars, self.chunk_overlap)
                wvecs = list(self._embed(wchunks)) if wchunks else []
                sim = self._best_sim(citation.passage, wchunks, wvecs)
                if sim >= self.threshold:
                    return (PassageMatch.FOUND, f"section+semantic:{sim:.2f}")
                return (PassageMatch.NOT_FOUND, f"section-mismatch:{sim:.2f}")
            return (PassageMatch.FOUND, "section-id")

        if citation.passage:
            chunks, vecs = self._embedded_chunks(corpus)
            sim = self._best_sim(citation.passage, chunks, vecs)
            if sim >= self.threshold:
                return (PassageMatch.FOUND, f"semantic:{sim:.2f}")
            return (PassageMatch.NOT_FOUND, f"none:{sim:.2f}")

        return (PassageMatch.NOT_FOUND, "none")
