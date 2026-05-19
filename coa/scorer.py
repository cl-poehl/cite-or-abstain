"""Run-level scoring: turn per-case categorizations + verifications into a report.

The headline metric (from the cite-or-abstain essay):

    score = coverage(cited-correct) - lambda * rate(uncited-confident-incorrect)

with lambda set by the cost of failure in the target domain. The default
lambda = 5.0 reflects a clinical-cost setting where one confidently-wrong
recommendation outweighs ~five correctly-cited ones.
"""
from __future__ import annotations

from collections import Counter

from .categorizer import categorize
from .llm.base import LLMBackend
from .types import Case, CaseScore, Category, RunReport
from .verifier import verify_citation


def score_case(case: Case, backend: LLMBackend, corpus: str | None = None) -> CaseScore:
    """Score a single case: categorize, then verify any cited claims.

    If the categorizer returns CITED, every extracted citation runs through the
    verifier (passage match + topical alignment). For non-cited outputs, no
    verification is performed.
    """
    categorization = categorize(case.output, backend)

    verifications = []
    if categorization.category == Category.CITED:
        for cit in categorization.citations:
            v = verify_citation(case.prompt, cit, corpus, backend)
            verifications.append(v)

    return CaseScore(
        case_id=case.id,
        category=categorization.category,
        citations=categorization.citations,
        verifications=verifications,
        rationale=categorization.rationale,
        expected_category=case.expected_category,
        expected_correct=case.expected_correct,
    )


def _is_cited_correct(cs: CaseScore) -> bool:
    """A cited case counts as 'correct' if:
      - expected_correct is True, OR (when not specified) the citation passed verification.
    """
    if cs.category != Category.CITED:
        return False
    if cs.expected_correct is True:
        return True
    if cs.expected_correct is False:
        return False
    # No ground-truth provided: fall back to verifier signals.
    if not cs.verifications:
        return False
    return all(v.passage_found and v.topical_alignment.value == "supports" for v in cs.verifications)


def _is_uncited_confident_incorrect(cs: CaseScore) -> bool:
    """A uncited-confident case counts as 'incorrect' if:
      - expected_correct is False, OR (when not specified) treat all uncited-confident as failure.

    The conservative default reflects the methodology: an uncited-confident clinical
    claim is treated as a failure regardless of whether the underlying fact happens
    to be correct, because the system has no way to verify it.
    """
    if cs.category != Category.UNCITED_CONFIDENT:
        return False
    if cs.expected_correct is False:
        return True
    if cs.expected_correct is True:
        return False
    # No ground-truth: conservative default — uncited-confident is treated as failure.
    return True


def compile_report(
    case_scores: list[CaseScore],
    model_name: str,
    lambda_: float = 5.0,
) -> RunReport:
    """Aggregate per-case scores into a run report with the headline metric."""
    n = len(case_scores)
    if n == 0:
        return RunReport(
            model=model_name,
            cases=[],
            counts={},
            coverage_cited_correct=0.0,
            rate_uncited_confident_incorrect=0.0,
            categorizer_accuracy=None,
            lambda_=lambda_,
            score=0.0,
        )

    cited_correct = sum(1 for cs in case_scores if _is_cited_correct(cs))
    uncited_confident_incorrect = sum(
        1 for cs in case_scores if _is_uncited_confident_incorrect(cs)
    )
    counts = Counter(cs.category.value for cs in case_scores)

    coverage = cited_correct / n
    rate_failure = uncited_confident_incorrect / n
    final_score = coverage - lambda_ * rate_failure

    # Categorizer accuracy only computable when expected_category is set on cases.
    labeled = [cs for cs in case_scores if cs.expected_category is not None]
    cat_accuracy: float | None = None
    if labeled:
        correct = sum(1 for cs in labeled if cs.correctly_categorized)
        cat_accuracy = correct / len(labeled)

    return RunReport(
        model=model_name,
        cases=case_scores,
        counts=dict(counts),
        coverage_cited_correct=coverage,
        rate_uncited_confident_incorrect=rate_failure,
        categorizer_accuracy=cat_accuracy,
        lambda_=lambda_,
        score=final_score,
    )
