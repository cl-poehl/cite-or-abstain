"""Golden test: the compiled report's stable fields must not drift unexpectedly.

Builds a fixed set of CaseScores covering every status and verdict (no LLM), compiles
a report, strips the churny free-text fields (rationales), and compares the rest to a
committed golden. Regenerate intentionally with:

    pytest tests/test_golden_report.py --update-goldens
"""
import json
from pathlib import Path

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

GOLDEN = Path(__file__).parent / "golden" / "report.json"


def _vr(match: PassageMatch, align: TopicalAlignment) -> VerificationResult:
    return VerificationResult(
        citation=Citation(source="EAU 2024", section="§6.4.2"),
        passage_match=match,
        topical_alignment=align,
        match_method="section-id" if match == PassageMatch.FOUND else "none",
    )


def _fixture_scores() -> list[CaseScore]:
    """One case per (status, verdict) combination worth pinning."""
    return [
        CaseScore(case_id="cited-verified", status=CaseStatus.SCORED, category=Category.CITED,
                  verifications=[_vr(PassageMatch.FOUND, TopicalAlignment.SUPPORTS)]),
        CaseScore(case_id="cited-miscited", status=CaseStatus.SCORED, category=Category.CITED,
                  verifications=[_vr(PassageMatch.FOUND, TopicalAlignment.CONTRADICTS)]),
        CaseScore(case_id="cited-fabricated", status=CaseStatus.SCORED, category=Category.CITED,
                  verifications=[_vr(PassageMatch.NOT_FOUND, TopicalAlignment.UNCERTAIN)]),
        CaseScore(case_id="uncited-confident", status=CaseStatus.SCORED,
                  category=Category.UNCITED_CONFIDENT),
        CaseScore(case_id="uncited-hedged", status=CaseStatus.SCORED,
                  category=Category.UNCITED_HEDGED),
        CaseScore(case_id="abstained", status=CaseStatus.SCORED, category=Category.ABSTAINED),
        CaseScore(case_id="invalid", status=CaseStatus.INVALID_OUTPUT, category=None),
        CaseScore(case_id="judge-failed", status=CaseStatus.JUDGE_FAILED, category=None),
        CaseScore(case_id="errored", status=CaseStatus.ERROR, category=None),
    ]


def _canonical(report) -> dict:
    """Report as a plain dict with churny free-text fields removed."""
    data = report.model_dump(mode="json")
    for case in data["cases"]:
        case.pop("rationale", None)
        for v in case.get("verifications", []):
            v.pop("verifier_rationale", None)
    return data


def test_report_matches_golden(update_goldens):
    report = compile_report(_fixture_scores(), "golden-model", lambda_=5.0)
    canonical = _canonical(report)

    if update_goldens:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(canonical, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    assert GOLDEN.exists(), "golden missing — run: pytest --update-goldens"
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert canonical == expected
