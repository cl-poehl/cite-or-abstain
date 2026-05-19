"""Verifier passage-matching tests — run without LLM keys (only the local matcher)."""
from coa.types import Citation
from coa.verifier import passage_in_corpus

CORPUS = """
EAU Guidelines on Prostate Cancer 2024.

§6.4.2 — Systemic therapy at metastatic diagnosis.
Intensified systemic therapy at the time of metastatic diagnosis improves
overall survival compared with ADT monotherapy.
"""


def test_passage_substring_match():
    cit = Citation(
        source="EAU 2024",
        section="§6.4.2",
        passage="Intensified systemic therapy at the time of metastatic diagnosis",
    )
    assert passage_in_corpus(cit, CORPUS) is True


def test_section_match():
    cit = Citation(source="EAU 2024", section="§6.4.2", passage="")
    assert passage_in_corpus(cit, CORPUS) is True


def test_token_overlap_match():
    cit = Citation(
        source="EAU 2024",
        section="",
        # Paraphrased but high token overlap with §6.4.2
        passage="intensified systemic therapy at metastatic diagnosis improves overall survival",
    )
    assert passage_in_corpus(cit, CORPUS, token_threshold=0.5) is True


def test_no_match_for_fabricated_passage():
    cit = Citation(
        source="EAU 2024",
        section="§99.9.9",
        passage="Quantum entanglement modulates androgen receptor signaling.",
    )
    assert passage_in_corpus(cit, CORPUS) is False


def test_empty_corpus_returns_false():
    cit = Citation(source="EAU 2024", section="§6.4.2", passage="foo")
    assert passage_in_corpus(cit, "") is False
