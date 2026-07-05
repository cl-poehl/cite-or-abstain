"""Determinism guarantees: idempotent verification + lossless serialization roundtrips."""
from coa.scorer import compile_report
from coa.testing import FixedBackend
from coa.types import (
    CaseScore,
    CaseStatus,
    Category,
    Citation,
    PassageMatch,
    TopicalAlignment,
    VerificationResult,
)
from coa.verifier import locate_passage, verify_citation

CORPUS = "§6.4.2 Intensified systemic therapy improves overall survival in mHSPC."


def test_locate_passage_is_idempotent():
    cit = Citation(source="X", section="§6.4.2", passage="Intensified systemic therapy")
    assert locate_passage(cit, CORPUS) == locate_passage(cit, CORPUS)


def test_verify_citation_is_idempotent():
    cit = Citation(source="X", section="§6.4.2", passage="Intensified systemic therapy")
    backend = FixedBackend("supports")
    a = verify_citation("claim", cit, CORPUS, backend)
    b = verify_citation("claim", cit, CORPUS, backend)
    assert a.model_dump() == b.model_dump()


def test_match_method_is_recorded():
    cit = Citation(source="X", section="§6.4.2")
    result = verify_citation("claim", cit, CORPUS, FixedBackend("supports"))
    assert result.match_method == "section-id"
    assert result.passage_match == PassageMatch.FOUND


def test_verification_result_roundtrips():
    vr = VerificationResult(
        citation=Citation(source="X", section="§1"),
        passage_match=PassageMatch.FOUND,
        topical_alignment=TopicalAlignment.SUPPORTS,
        match_method="substring",
    )
    restored = VerificationResult.model_validate_json(vr.model_dump_json())
    assert restored.model_dump() == vr.model_dump()
    assert restored.verdict == vr.verdict


def test_run_report_roundtrips_losslessly():
    scores = [
        CaseScore(
            case_id="a",
            status=CaseStatus.SCORED,
            category=Category.CITED,
            citations=[Citation(source="X", section="§1")],
            verifications=[
                VerificationResult(
                    citation=Citation(source="X", section="§1"),
                    passage_match=PassageMatch.FOUND,
                    topical_alignment=TopicalAlignment.SUPPORTS,
                    match_method="substring",
                )
            ],
            expected_correct=True,
        ),
        CaseScore(case_id="b", status=CaseStatus.JUDGE_FAILED, category=None),
    ]
    from coa.scorer import frozen_judge_fingerprint
    from coa.types import RunReport

    report = compile_report(
        scores, "m", frozen_judge=frozen_judge_fingerprint("m", 1), corpus={"corpus_id": "x"}
    )
    restored = RunReport.model_validate_json(report.model_dump_json())
    assert restored.model_dump_json() == report.model_dump_json()
