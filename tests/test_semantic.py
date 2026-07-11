"""Semantic matcher: logic + existence-verifying semantics, with a fake offline embedder."""
from __future__ import annotations

from coa.semantic import SemanticMatcher, _cosine
from coa.types import Citation, PassageMatch

CORPUS = (
    "§4.2.1 — Active surveillance criteria\n"
    "Active surveillance is the preferred approach for ISUP grade group 1 disease.\n\n"
    "§6.4.2 — Systemic therapy\n"
    "Intensified systemic therapy at metastatic diagnosis improves overall survival.\n"
)


def _bow_embedder(dim: int = 64):
    """Deterministic hashing bag-of-words embedder — no network, stable across runs.

    Paraphrases that share content words land close in cosine space; unrelated text does
    not. Good enough to exercise the matcher's control flow and thresholds deterministically.
    """
    def embed(texts):
        vecs = []
        for t in texts:
            v = [0.0] * dim
            for w in t.lower().split():
                v[hash(w) % dim] += 1.0
            vecs.append(v)
        return vecs
    return embed


def test_cosine_basic():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert _cosine([1, 0], [0, 1]) == 0.0
    assert _cosine([0, 0], [1, 1]) == 0.0  # zero vector -> 0, no div-by-zero


def test_no_corpus_is_unverifiable():
    m = SemanticMatcher(_bow_embedder(), threshold=0.5)
    match, how = m(Citation(source="x", passage="anything"), None)
    assert match == PassageMatch.UNVERIFIABLE and how == "no-corpus"


def test_paraphrase_located_whole_corpus():
    # A paraphrase of the §6.4.2 line, no section given -> whole-corpus semantic search.
    m = SemanticMatcher(_bow_embedder(), threshold=0.4)
    cit = Citation(source="EAU", section="",
                   passage="Intensified systemic therapy improves survival at metastatic diagnosis")
    match, how = m(cit, CORPUS)
    assert match == PassageMatch.FOUND
    assert how.startswith("semantic:")


def test_fabricated_section_not_resurrected_by_similarity():
    # Section absent from corpus -> NOT_FOUND(section-absent), even though the passage
    # is semantically identical to real prose elsewhere. Fabrication must survive.
    m = SemanticMatcher(_bow_embedder(), threshold=0.0)  # threshold 0 => sim always passes
    cit = Citation(source="EAU", section="99.9",
                   passage="Intensified systemic therapy improves overall survival")
    match, how = m(cit, CORPUS)
    assert match == PassageMatch.NOT_FOUND and how == "section-absent"


def test_unrelated_passage_not_found():
    m = SemanticMatcher(_bow_embedder(), threshold=0.5)
    cit = Citation(source="EAU", section="",
                   passage="the quarterly revenue forecast for the fiscal year")
    match, how = m(cit, CORPUS)
    assert match == PassageMatch.NOT_FOUND and how.startswith("none:")


def test_drop_in_signature_matches_locate_passage():
    # Must be callable as (citation, corpus, fuzzy_threshold) like locate_passage.
    from coa.testing import FixedBackend  # existing test stub
    from coa.verifier import verify_citation

    m = SemanticMatcher(_bow_embedder(), threshold=0.4)
    cit = Citation(source="EAU", section="",
                   passage="Intensified systemic therapy improves survival at metastatic diagnosis")
    res = verify_citation("claim", cit, CORPUS, FixedBackend("supports"), locate=m)
    assert res.passage_match == PassageMatch.FOUND
