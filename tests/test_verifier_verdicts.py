"""Verifier verdict + three-state passage-match tests — mostly no LLM needed."""
from coa.testing import FixedBackend
from coa.types import (
    Citation,
    CitationVerdict,
    PassageMatch,
    TopicalAlignment,
    VerificationResult,
)
from coa.verifier import locate_passage, verify_citation

CORPUS = """
EAU Guidelines on Prostate Cancer 2024.

§6.4.2 — Systemic therapy at metastatic diagnosis.
Intensified systemic therapy at the time of metastatic diagnosis improves
overall survival compared with ADT monotherapy.
"""


def test_locate_reports_matched_tier():
    assert locate_passage(
        Citation(source="X", passage="Intensified systemic therapy at the time of metastatic"),
        CORPUS,
    ) == (PassageMatch.FOUND, "substring")
    assert locate_passage(Citation(source="X", section="§6.4.2"), CORPUS) == (
        PassageMatch.FOUND,
        "section-id",
    )
    assert locate_passage(Citation(source="X", passage="foo bar baz nonexistent quux"), CORPUS) == (
        PassageMatch.NOT_FOUND,
        "none",
    )


def test_locate_no_corpus_is_unverifiable_not_notfound():
    match, method = locate_passage(Citation(source="X", section="§6.4.2"), None)
    assert match == PassageMatch.UNVERIFIABLE
    assert method == "no-corpus"


def test_verdict_mapping():
    def vr(match, align, method=""):
        return VerificationResult(
            citation=Citation(source="X"),
            passage_match=match,
            topical_alignment=align,
            match_method=method,
        )

    assert vr(PassageMatch.FOUND, TopicalAlignment.SUPPORTS).verdict == CitationVerdict.VERIFIED
    assert vr(PassageMatch.FOUND, TopicalAlignment.CONTRADICTS).verdict == CitationVerdict.MISCITED
    assert vr(PassageMatch.FOUND, TopicalAlignment.UNRELATED).verdict == CitationVerdict.MISCITED
    assert vr(PassageMatch.FOUND, TopicalAlignment.UNCERTAIN).verdict == CitationVerdict.UNCERTAIN
    assert (
        vr(PassageMatch.UNVERIFIABLE, TopicalAlignment.UNCERTAIN).verdict
        == CitationVerdict.UNVERIFIABLE
    )


def test_not_found_verdict_splits_by_cause():
    """NOT_FOUND has three causes; only `section-absent` is evidence of fabrication."""

    def nf(method):
        return VerificationResult(
            citation=Citation(source="X"),
            passage_match=PassageMatch.NOT_FOUND,
            topical_alignment=TopicalAlignment.SUPPORTS,
            match_method=method,
        ).verdict

    # The cited section id is not in the corpus -> positive evidence of absence.
    assert nf("section-absent") == CitationVerdict.FABRICATED
    # Section is real, passage attributed elsewhere -> a wrong-location citation.
    assert nf("section-mismatch") == CitationVerdict.MISCITED
    # No section and the matcher simply could not find it -> inconclusive, NOT fabrication.
    assert nf("none") == CitationVerdict.UNLOCATED
    # Unknown/absent method must fail safe to the non-accusatory verdict.
    assert nf("") == CitationVerdict.UNLOCATED


def test_verify_citation_skips_judge_when_not_found():
    """A fabricated citation must not consult the alignment judge; verdict is FABRICATED."""
    backend = FixedBackend("supports")  # would say 'supports' if called
    result = verify_citation(
        "Some claim",
        Citation(source="X", section="§99.9", passage="wholly invented passage nowhere in corpus"),
        CORPUS,
        backend,
    )
    assert result.passage_match == PassageMatch.NOT_FOUND
    assert result.verdict == CitationVerdict.FABRICATED


def test_verify_citation_found_then_judged():
    backend = FixedBackend("supports")
    result = verify_citation(
        "Intensified therapy improves survival",
        Citation(source="X", section="§6.4.2", passage="Intensified systemic therapy at the time"),
        CORPUS,
        backend,
    )
    assert result.passage_match == PassageMatch.FOUND
    assert result.verdict == CitationVerdict.VERIFIED


def test_verify_citation_no_corpus_is_unverifiable():
    backend = FixedBackend("supports")
    result = verify_citation("claim", Citation(source="X", section="§6.4.2"), None, backend)
    assert result.verdict == CitationVerdict.UNVERIFIABLE
