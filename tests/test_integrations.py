"""DeepEval adapter tests — the pure mapping + the missing-dependency guard."""
import pytest

from coa.integrations.deepeval import (
    _HAVE_DEEPEVAL,
    CiteOrAbstainMetric,
    casescore_to_result,
    evaluate_output,
)
from coa.testing import RoutedBackend
from coa.types import (
    CaseScore,
    CaseStatus,
    Category,
    Citation,
    PassageMatch,
    TopicalAlignment,
    VerificationResult,
)


def _cs(status=CaseStatus.SCORED, category=None, verifications=None, expected_correct=None):
    return CaseScore(
        case_id="x",
        status=status,
        category=category,
        verifications=verifications or [],
        expected_correct=expected_correct,
    )


def test_abstained_passes():
    score, passed, reason = casescore_to_result(_cs(category=Category.ABSTAINED))
    assert passed is True and score == 1.0
    assert "abstained" in reason


def test_verified_cited_passes():
    v = VerificationResult(
        citation=Citation(source="X"),
        passage_match=PassageMatch.FOUND,
        topical_alignment=TopicalAlignment.SUPPORTS,
    )
    score, passed, _ = casescore_to_result(_cs(category=Category.CITED, verifications=[v]))
    assert passed is True and score == 1.0


def test_uncited_confident_fails():
    score, passed, _ = casescore_to_result(_cs(category=Category.UNCITED_CONFIDENT))
    assert passed is False and score == 0.0


def test_fabricated_citation_fails():
    v = VerificationResult(
        citation=Citation(source="X", section="§9.99"),
        passage_match=PassageMatch.NOT_FOUND,
        topical_alignment=TopicalAlignment.UNCERTAIN,
        match_method="section-absent",
    )
    score, passed, reason = casescore_to_result(_cs(category=Category.CITED, verifications=[v]))
    assert passed is False and score == 0.0
    assert "fabricated" in reason


def test_unlocated_citation_fails_but_is_not_called_fabricated():
    """Still fails (penalized), but the reason must not overclaim fabrication."""
    v = VerificationResult(
        citation=Citation(source="X", passage="paraphrased"),
        passage_match=PassageMatch.NOT_FOUND,
        topical_alignment=TopicalAlignment.UNCERTAIN,
        match_method="none",
    )
    score, passed, reason = casescore_to_result(_cs(category=Category.CITED, verifications=[v]))
    assert passed is False and score == 0.0
    assert "unlocated" in reason and "fabricated" not in reason


def test_invalid_and_judge_failed_fail():
    assert casescore_to_result(_cs(status=CaseStatus.INVALID_OUTPUT))[1] is False
    assert casescore_to_result(_cs(status=CaseStatus.JUDGE_FAILED))[1] is False


def test_evaluate_output_end_to_end_with_fake_backend():
    backend = RoutedBackend(lambda s, u: '{"category":"abstained","rationale":"r"}')
    cs = evaluate_output("prompt?", "I cannot recommend without more data.", backend)
    assert cs.category == Category.ABSTAINED
    score, passed, _ = casescore_to_result(cs)
    assert passed is True


@pytest.mark.skipif(_HAVE_DEEPEVAL, reason="deepeval is installed")
def test_metric_raises_without_deepeval():
    with pytest.raises(ImportError):
        CiteOrAbstainMetric(RoutedBackend(lambda s, u: "x"))
