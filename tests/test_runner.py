"""Crash-isolating batch runner + judge-vs-mechanical delta."""
from coa.scorer import compile_report, score_cases
from coa.testing import RoutedBackend
from coa.types import (
    Case,
    CaseScore,
    CaseStatus,
    Category,
    Citation,
    PassageMatch,
    TopicalAlignment,
    VerificationResult,
)


def test_runner_isolates_one_crashing_case_and_runs_the_rest():
    def route(system, user):
        if "BOOM" in user:
            raise RuntimeError("simulated backend failure")
        return '{"category":"abstained","rationale":"r"}'

    backend = RoutedBackend(route)
    cases = [
        Case(id="ok1", prompt="q", output="a genuine answer"),
        Case(id="boom", prompt="q", output="this one says BOOM"),
        Case(id="ok2", prompt="q", output="another genuine answer"),
    ]
    scores = score_cases(cases, backend)

    by_id = {cs.case_id: cs for cs in scores}
    assert by_id["boom"].status == CaseStatus.ERROR
    assert by_id["ok1"].status == CaseStatus.SCORED
    assert by_id["ok2"].status == CaseStatus.SCORED
    # scored + error == total: the batch always completes.
    assert len(scores) == len(cases)


def test_errored_case_excluded_from_denominator():
    scores = [
        CaseScore(case_id="a", status=CaseStatus.SCORED, category=Category.CITED,
                  expected_correct=True),
        CaseScore(case_id="b", status=CaseStatus.ERROR, category=None),
    ]
    report = compile_report(scores, "m")
    assert report.scored_denominator == 1
    assert report.coverage_cited_correct == 1.0
    assert abs(report.error_rate - 0.5) < 1e-9
    assert report.status_counts["error"] == 1


def test_judge_vs_mechanical_delta_counts_found_but_unsupported():
    """A citation FOUND but not SUPPORTS is where the judge beats a naive string match."""
    v_found_unsupported = VerificationResult(
        citation=Citation(source="X"),
        passage_match=PassageMatch.FOUND,
        topical_alignment=TopicalAlignment.CONTRADICTS,
    )
    v_verified = VerificationResult(
        citation=Citation(source="Y"),
        passage_match=PassageMatch.FOUND,
        topical_alignment=TopicalAlignment.SUPPORTS,
    )
    v_fabricated = VerificationResult(
        citation=Citation(source="Z"),
        passage_match=PassageMatch.NOT_FOUND,
        topical_alignment=TopicalAlignment.UNCERTAIN,
    )
    scores = [
        CaseScore(
            case_id="a",
            status=CaseStatus.SCORED,
            category=Category.CITED,
            verifications=[v_found_unsupported, v_verified, v_fabricated],
        )
    ]
    report = compile_report(scores, "m")
    # Only v_found_unsupported: FOUND by string match, but the judge would not pass it.
    assert report.judge_vs_mechanical_delta == 1
