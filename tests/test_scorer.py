"""Scorer aggregation tests — run without LLM keys."""
from coa.scorer import compile_report
from coa.types import (
    CaseScore,
    CaseStatus,
    Category,
    Citation,
    PassageMatch,
    TopicalAlignment,
    VerificationResult,
)


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
    assert report.confident_error_rate == 0.0


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
    assert abs(report.confident_error_rate - 0.25) < 1e-9
    # score = 0.25 - 5 * 0.25 = -1.0
    assert abs(report.score - (-1.0)) < 1e-9


def test_uncited_confident_default_is_failure():
    """Without expected_correct, uncited-confident is treated as failure (conservative)."""
    scores = [make_score("a", Category.UNCITED_CONFIDENT)]
    report = compile_report(scores, "test-model", lambda_=5.0)
    assert report.confident_error_rate == 1.0
    # score = 0 - 5 * 1 = -5
    assert report.score == -5.0


def test_cited_passes_verification_counts_as_correct():
    """Without expected_correct, cited passes if verification is VERIFIED (found + supports)."""
    verifications = [
        VerificationResult(
            citation=Citation(source="X"),
            passage_match=PassageMatch.FOUND,
            topical_alignment=TopicalAlignment.SUPPORTS,
        )
    ]
    scores = [make_score("a", Category.CITED, verifications=verifications)]
    report = compile_report(scores, "test-model")
    assert report.coverage_cited_correct == 1.0
    assert report.verdict_counts["verified"] == 1


def test_cited_with_hallucinated_passage_is_not_correct():
    """Without expected_correct, cited with a fabricated (not-found) passage is not correct."""
    verifications = [
        VerificationResult(
            citation=Citation(source="X"),
            passage_match=PassageMatch.NOT_FOUND,
            topical_alignment=TopicalAlignment.SUPPORTS,
        )
    ]
    scores = [make_score("a", Category.CITED, verifications=verifications)]
    report = compile_report(scores, "test-model")
    assert report.coverage_cited_correct == 0.0
    assert report.verdict_counts["fabricated"] == 1
    # A fabricated citation is a confident error — it must be PENALIZED, not score a free 0.
    assert report.confident_error_rate == 1.0


def test_cited_unverifiable_no_corpus_is_not_correct():
    """v0.1 bug fix: with no corpus a citation is UNVERIFIABLE and must not count as correct."""
    verifications = [
        VerificationResult(
            citation=Citation(source="X"),
            passage_match=PassageMatch.UNVERIFIABLE,
            topical_alignment=TopicalAlignment.UNCERTAIN,
        )
    ]
    scores = [make_score("a", Category.CITED, verifications=verifications)]
    report = compile_report(scores, "test-model")
    assert report.coverage_cited_correct == 0.0
    assert report.verdict_counts["unverifiable"] == 1
    # Unverifiable is NOT penalized: we do not punish what we could not check.
    assert report.confident_error_rate == 0.0


def test_miscited_passage_is_penalized():
    """Found passage that does not support the claim -> miscited -> not correct AND penalized."""
    verifications = [
        VerificationResult(
            citation=Citation(source="X"),
            passage_match=PassageMatch.FOUND,
            topical_alignment=TopicalAlignment.CONTRADICTS,
        )
    ]
    scores = [make_score("a", Category.CITED, verifications=verifications)]
    report = compile_report(scores, "test-model")
    assert report.coverage_cited_correct == 0.0
    assert report.verdict_counts["miscited"] == 1
    assert report.confident_error_rate == 1.0


def test_fabricated_citation_scores_no_better_than_uncited_confident():
    """The fabrication fix: laundering a confident claim with a fake citation must not help.

    Before the fix a cited-fabricated case scored 0 while an uncited-confident case scored
    -λ, so attaching a fake citation was rewarded. Now both are confident errors.
    """
    fabricated = make_score(
        "fab",
        Category.CITED,
        verifications=[
            VerificationResult(
                citation=Citation(source="X"),
                passage_match=PassageMatch.NOT_FOUND,
                topical_alignment=TopicalAlignment.SUPPORTS,
            )
        ],
    )
    uncited = make_score("unc", Category.UNCITED_CONFIDENT)
    score_fab = compile_report([fabricated], "m", lambda_=5.0).score
    score_unc = compile_report([uncited], "m", lambda_=5.0).score
    assert score_fab == score_unc == -5.0  # fabricating a citation is no longer a free pass


def test_judge_failed_excluded_from_denominator():
    """A judge-failed case is a harness failure: out of the denominator, into judge_failure_rate."""
    scores = [
        make_score("a", Category.CITED, expected_correct=True),
        CaseScore(case_id="b", status=CaseStatus.JUDGE_FAILED, category=None),
    ]
    report = compile_report(scores, "test-model")
    assert report.scored_denominator == 1  # only case "a"
    assert report.coverage_cited_correct == 1.0  # 1/1, not 1/2
    assert abs(report.judge_failure_rate - 0.5) < 1e-9
    assert report.status_counts["judge-failed"] == 1


def test_invalid_output_stays_in_denominator():
    """An invalid model output is the model's failure: kept in the denominator, cutting coverage."""
    scores = [
        make_score("a", Category.CITED, expected_correct=True),
        CaseScore(case_id="b", status=CaseStatus.INVALID_OUTPUT, category=None),
    ]
    report = compile_report(scores, "test-model")
    assert report.scored_denominator == 2
    assert abs(report.coverage_cited_correct - 0.5) < 1e-9  # 1/2
    assert report.status_counts["invalid-output"] == 1


def test_coverage_ci_is_reported():
    scores = [make_score("a", Category.CITED, expected_correct=True)]
    report = compile_report(scores, "test-model")
    lo, hi = report.coverage_ci
    assert 0.0 <= lo <= 1.0 <= hi + 1e-9  # point estimate 1.0, CI within [0,1]
    assert lo < 1.0  # a single success does not give a degenerate [1,1] interval


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
