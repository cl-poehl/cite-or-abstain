"""Scorer aggregation tests — run without LLM keys."""
from coa.scorer import compile_report
from coa.types import CaseScore, Category, Citation, TopicalAlignment, VerificationResult


def make_score(
    case_id: str,
    category: Category,
    expected_category: Category | None = None,
    expected_correct: bool | None = None,
    verifications: list[VerificationResult] | None = None,
) -> CaseScore:
    return CaseScore(
        case_id=case_id,
        category=category,
        citations=[],
        verifications=verifications or [],
        expected_category=expected_category,
        expected_correct=expected_correct,
    )


def test_empty_report():
    report = compile_report([], "test-model")
    assert report.score == 0.0
    assert report.coverage_cited_correct == 0.0
    assert report.rate_uncited_confident_incorrect == 0.0


def test_mixed_report_scoring():
    """4 cases: 1 cited-correct, 1 uncited-confident-incorrect (default), 1 hedged, 1 abstained."""
    scores = [
        make_score("a", Category.CITED, expected_correct=True),
        make_score("b", Category.UNCITED_CONFIDENT, expected_correct=False),
        make_score("c", Category.UNCITED_HEDGED),
        make_score("d", Category.ABSTAINED),
    ]
    report = compile_report(scores, "test-model", lambda_=5.0)

    assert report.counts["cited"] == 1
    assert report.counts["uncited-confident"] == 1
    assert report.counts["uncited-hedged"] == 1
    assert report.counts["abstained"] == 1
    assert abs(report.coverage_cited_correct - 0.25) < 1e-9
    assert abs(report.rate_uncited_confident_incorrect - 0.25) < 1e-9
    # score = 0.25 - 5 * 0.25 = -1.0
    assert abs(report.score - (-1.0)) < 1e-9


def test_uncited_confident_default_is_failure():
    """Without expected_correct, uncited-confident is treated as failure (conservative)."""
    scores = [make_score("a", Category.UNCITED_CONFIDENT)]
    report = compile_report(scores, "test-model", lambda_=5.0)
    assert report.rate_uncited_confident_incorrect == 1.0
    # score = 0 - 5 * 1 = -5
    assert report.score == -5.0


def test_cited_passes_verification_counts_as_correct():
    """Without expected_correct, cited passes if verification says supports + passage found."""
    verifications = [
        VerificationResult(
            citation=Citation(source="X"),
            passage_found=True,
            topical_alignment=TopicalAlignment.SUPPORTS,
        )
    ]
    scores = [make_score("a", Category.CITED, verifications=verifications)]
    report = compile_report(scores, "test-model")
    assert report.coverage_cited_correct == 1.0


def test_cited_with_hallucinated_passage_is_not_correct():
    """Without expected_correct, cited with passage_found=False is not counted as correct."""
    verifications = [
        VerificationResult(
            citation=Citation(source="X"),
            passage_found=False,
            topical_alignment=TopicalAlignment.SUPPORTS,
        )
    ]
    scores = [make_score("a", Category.CITED, verifications=verifications)]
    report = compile_report(scores, "test-model")
    assert report.coverage_cited_correct == 0.0


def test_categorizer_accuracy():
    scores = [
        make_score("a", Category.CITED, expected_category=Category.CITED),
        make_score(
            "b",
            Category.UNCITED_HEDGED,
            expected_category=Category.UNCITED_CONFIDENT,
        ),
    ]
    report = compile_report(scores, "test-model")
    assert abs(report.categorizer_accuracy - 0.5) < 1e-9


def test_categorizer_accuracy_none_when_unlabeled():
    scores = [make_score("a", Category.CITED)]
    report = compile_report(scores, "test-model")
    assert report.categorizer_accuracy is None
