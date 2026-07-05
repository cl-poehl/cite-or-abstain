"""Validate the categorizer — the harness's own LLM judge — against human labels.

The whole methodology insists you measure a judge before trusting it (docs/FINDINGS.md
§3, §6). The categorizer *is* an LLM judge, so this module turns its predictions on a
labeled set into the numbers that decide whether to trust it: accuracy with a Wilson
interval, Gwet's AC1 (robust to the skew clinical labels always have), and a confusion
matrix showing *where* it errs.

It operates offline on a scored `RunReport` whose cases carry `expected_category`
labels — no backend needed. Producing a meaningful number is on you: label a real,
held-out set and run `coa score` over it first. A high accuracy on 8 toy cases means
nothing; that is the point.
"""
from __future__ import annotations

from pydantic import BaseModel

from .stats import gwet_ac1, wilson_ci
from .types import CaseStatus, RunReport


class JudgeValidation(BaseModel):
    """The categorizer's reliability against human-labeled expected categories."""

    n_labeled: int
    accuracy: float | None = None
    accuracy_ci: tuple[float, float] = (0.0, 0.0)
    gwet_ac1: float | None = None
    # confusion[expected][predicted] = count
    confusion: dict[str, dict[str, int]] = {}


def validate_judge(report: RunReport) -> JudgeValidation:
    """Compute categorizer accuracy + Gwet AC1 + confusion from a scored, labeled report."""
    pairs = [
        (cs.category.value, cs.expected_category.value)
        for cs in report.cases
        if cs.status == CaseStatus.SCORED
        and cs.category is not None
        and cs.expected_category is not None
    ]
    n = len(pairs)
    if n == 0:
        return JudgeValidation(n_labeled=0)

    agree = sum(1 for pred, exp in pairs if pred == exp)
    confusion: dict[str, dict[str, int]] = {}
    for pred, exp in pairs:
        confusion.setdefault(exp, {})
        confusion[exp][pred] = confusion[exp].get(pred, 0) + 1

    return JudgeValidation(
        n_labeled=n,
        accuracy=agree / n,
        accuracy_ci=wilson_ci(agree, n),
        gwet_ac1=gwet_ac1(pairs),
        confusion=confusion,
    )
