"""Verifier passage-matching tests — run without LLM keys (only the local matcher)."""
from coa.types import Citation, PassageMatch
from coa.verifier import locate_passage, passage_in_corpus

CORPUS = """
EAU Guidelines on Prostate Cancer 2024.

§6.4.2 — Systemic therapy at metastatic diagnosis.
Intensified systemic therapy at the time of metastatic diagnosis improves
overall survival compared with ADT monotherapy.

§8.2.4 — High-risk localized disease.
Radical prostatectomy with extended pelvic lymph node dissection is a
recommended option for high-risk localized disease.
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


def test_fuzzy_paraphrase_match():
    cit = Citation(
        source="EAU 2024",
        section="",
        # Paraphrased but a close fuzzy match to §6.4.2's text.
        passage="intensified systemic therapy at metastatic diagnosis improves overall survival",
    )
    assert passage_in_corpus(cit, CORPUS) is True


def test_no_match_for_fabricated_passage():
    cit = Citation(
        source="EAU 2024",
        section="§99.9.9",
        passage="Quantum entanglement modulates androgen receptor signaling.",
    )
    assert passage_in_corpus(cit, CORPUS) is False


def test_section_absent_is_not_found():
    """A cited section that does not exist in the corpus is a fabricated reference."""
    cit = Citation(source="EAU 2024", section="§99.9.9", passage="anything at all here")
    match, method = locate_passage(cit, CORPUS)
    assert match == PassageMatch.NOT_FOUND
    assert method == "section-absent"


def test_section_mismatch_is_caught():
    """Real section id, but the passage is from a DIFFERENT section -> wrong-location citation.

    This is the class of miscitation a whole-corpus bag-of-words match is blind to: the
    §8.2.4 passage really exists, but it is attributed to §6.4.2.
    """
    cit = Citation(
        source="EAU 2024",
        section="§6.4.2",
        passage="Radical prostatectomy with extended pelvic lymph node dissection",
    )
    match, method = locate_passage(cit, CORPUS)
    assert match == PassageMatch.NOT_FOUND
    assert method == "section-mismatch"


def test_within_section_paraphrase_is_found():
    """A reordered/paraphrased passage whose content words are all in the cited section is
    found (via fuzzy or the section-scoped token tier) — not falsely called section-mismatch."""
    cit = Citation(
        source="EAU 2024",
        section="§6.4.2",
        passage="systemic therapy, intensified at metastatic diagnosis, improves survival overall",
    )
    match, method = locate_passage(cit, CORPUS)
    assert match == PassageMatch.FOUND
    assert method.startswith("section")


def test_section_and_passage_consistent_is_found():
    cit = Citation(
        source="EAU 2024",
        section="§6.4.2",
        passage="Intensified systemic therapy at the time of metastatic diagnosis",
    )
    match, method = locate_passage(cit, CORPUS)
    assert match == PassageMatch.FOUND
    assert method.startswith("section+")


def test_empty_corpus_returns_false():
    cit = Citation(source="EAU 2024", section="§6.4.2", passage="foo")
    assert passage_in_corpus(cit, "") is False
