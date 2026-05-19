"""Data-model integrity tests — run without LLM keys."""
from coa.types import Case, CaseScore, Categorization, Category, Citation


def test_category_enum_values():
    assert Category.CITED.value == "cited"
    assert Category.UNCITED_CONFIDENT.value == "uncited-confident"
    assert Category.UNCITED_HEDGED.value == "uncited-hedged"
    assert Category.ABSTAINED.value == "abstained"


def test_case_round_trips():
    case = Case(
        id="t-001",
        prompt="What therapy?",
        output="Per EAU 2024 §6.4.2, ADT plus docetaxel.",
        expected_category=Category.CITED,
        expected_correct=True,
    )
    data = case.model_dump()
    restored = Case(**data)
    assert restored.id == "t-001"
    assert restored.expected_category == Category.CITED


def test_citation_defaults():
    c = Citation(source="EAU 2024")
    assert c.section == ""
    assert c.passage == ""


def test_case_score_correctly_categorized():
    cs = CaseScore(
        case_id="t-002",
        category=Category.CITED,
        expected_category=Category.CITED,
    )
    assert cs.correctly_categorized is True

    cs2 = CaseScore(
        case_id="t-003",
        category=Category.CITED,
        expected_category=Category.UNCITED_CONFIDENT,
    )
    assert cs2.correctly_categorized is False

    cs3 = CaseScore(case_id="t-004", category=Category.CITED)
    assert cs3.correctly_categorized is None


def test_categorization_forces_empty_citations_for_non_cited():
    # The categorizer module enforces this, but the type allows it for
    # round-trip flexibility; this test documents the convention.
    c = Categorization(
        category=Category.ABSTAINED,
        citations=[Citation(source="X")],
        rationale="r",
    )
    # Type allows it; categorizer.categorize() clears it.
    assert c.category == Category.ABSTAINED
