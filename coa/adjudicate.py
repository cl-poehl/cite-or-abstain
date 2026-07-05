"""Select the subset of a scored run that a human should adjudicate.

The harness is a screen, not a verdict (see docs/FINDINGS.md §3). The production
pattern is: auto-score, then send the cases the automation is least trustworthy on —
plus a random audit sample — to a human. This module implements that selection,
deterministically and offline (it reads a saved `RunReport`; no backend needed).

A case is flagged for review when any of these hold:
  - its disposition is a harness or model failure (judge-failed / invalid-output / error);
  - it is `uncited-confident` (the highest-risk category);
  - any of its citations is not cleanly `verified` (miscited / fabricated / uncertain /
    unverifiable).
Everything else is clean, and a seeded random fraction of the clean cases is added as a
routine audit so systematic categorizer/verifier errors surface even when nothing
tripped a flag.
"""
from __future__ import annotations

import random

from pydantic import BaseModel

from .types import Case, CaseScore, CaseStatus, Category, CitationVerdict, RunReport

_REVIEW_STATUSES = {CaseStatus.JUDGE_FAILED, CaseStatus.INVALID_OUTPUT, CaseStatus.ERROR}


class AdjudicationItem(BaseModel):
    """One case routed to a human, with the reason(s) it was selected."""

    case_id: str
    reasons: list[str]
    category: str | None = None
    status: str = ""
    verdicts: list[str] = []
    prompt: str = ""
    output: str = ""


class Worklist(BaseModel):
    """The human-review worklist plus a summary of how it was built."""

    items: list[AdjudicationItem]
    total_cases: int
    flagged: int  # selected because a review rule fired
    sampled: int  # clean cases pulled in by the random audit
    sample_frac: float
    seed: int


def _flag_reasons(cs: CaseScore) -> list[str]:
    """Deterministic, deduplicated review reasons for a case (empty => clean)."""
    reasons: list[str] = []
    if cs.status in _REVIEW_STATUSES:
        reasons.append(f"status:{cs.status.value}")
    if cs.category == Category.UNCITED_CONFIDENT:
        reasons.append("uncited-confident (highest-risk)")
    for v in cs.verifications:
        if v.verdict != CitationVerdict.VERIFIED:
            r = f"verdict:{v.verdict.value}"
            if r not in reasons:
                reasons.append(r)
    return reasons


def build_worklist(
    report: RunReport,
    cases_by_id: dict[str, Case] | None = None,
    sample_frac: float = 0.1,
    seed: int = 0,
) -> Worklist:
    """Build the human-review worklist from a scored run.

    Args:
        report: a compiled RunReport (e.g. loaded from `coa score -o report.json`).
        cases_by_id: optional map id -> Case, to enrich items with prompt/output text.
        sample_frac: fraction of otherwise-clean cases to pull in as a random audit.
        seed: RNG seed for the random audit (reproducible worklists).
    """
    cases_by_id = cases_by_id or {}
    rng = random.Random(seed)
    items: list[AdjudicationItem] = []
    flagged = 0
    sampled = 0

    for cs in report.cases:
        reasons = _flag_reasons(cs)
        selected = bool(reasons)
        if not selected:
            # Draw for every clean case so the sample is deterministic given the seed.
            if rng.random() < sample_frac:
                selected = True
                reasons = ["random-audit"]
                sampled += 1
        else:
            flagged += 1

        if not selected:
            continue

        src = cases_by_id.get(cs.case_id)
        items.append(
            AdjudicationItem(
                case_id=cs.case_id,
                reasons=reasons,
                category=cs.category.value if cs.category else None,
                status=cs.status.value,
                verdicts=[v.verdict.value for v in cs.verifications],
                prompt=src.prompt if src else "",
                output=src.output if src else "",
            )
        )

    return Worklist(
        items=items,
        total_cases=len(report.cases),
        flagged=flagged,
        sampled=sampled,
        sample_frac=sample_frac,
        seed=seed,
    )
