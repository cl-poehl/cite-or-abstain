"""Adjudication worklist selection."""
from coa.adjudicate import build_worklist
from coa.scorer import compile_report
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


def _verified():
    return VerificationResult(
        citation=Citation(source="X"),
        passage_match=PassageMatch.FOUND,
        topical_alignment=TopicalAlignment.SUPPORTS,
    )


def _fabricated():
    return VerificationResult(
        citation=Citation(source="X"),
        passage_match=PassageMatch.NOT_FOUND,
        topical_alignment=TopicalAlignment.UNCERTAIN,
    )


def _report():
    scores = [
        CaseScore(case_id="clean", status=CaseStatus.SCORED, category=Category.CITED,
                  verifications=[_verified()]),
        CaseScore(case_id="confident", status=CaseStatus.SCORED,
                  category=Category.UNCITED_CONFIDENT),
        CaseScore(case_id="fab", status=CaseStatus.SCORED, category=Category.CITED,
                  verifications=[_fabricated()]),
        CaseScore(case_id="jf", status=CaseStatus.JUDGE_FAILED, category=None),
    ]
    return compile_report(scores, "m")


def test_flags_confident_fabricated_and_judgefailed_not_clean():
    wl = build_worklist(_report(), sample_frac=0.0)  # no random audit
    ids = {it.case_id for it in wl.items}
    assert "confident" in ids  # uncited-confident always reviewed
    assert "fab" in ids  # fabricated citation
    assert "jf" in ids  # judge-failed
    assert "clean" not in ids  # cleanly verified -> not flagged
    assert wl.flagged == 3
    assert wl.sampled == 0


def test_reasons_are_populated():
    wl = build_worklist(_report(), sample_frac=0.0)
    reasons = {it.case_id: it.reasons for it in wl.items}
    assert any("uncited-confident" in r for r in reasons["confident"])
    assert any("fabricated" in r for r in reasons["fab"])
    assert any("judge-failed" in r for r in reasons["jf"])


def test_random_audit_is_seed_deterministic():
    rep = _report()
    a = build_worklist(rep, sample_frac=1.0, seed=7)  # pull in every clean case
    b = build_worklist(rep, sample_frac=1.0, seed=7)
    assert [i.case_id for i in a.items] == [i.case_id for i in b.items]
    # sample_frac=1.0 pulls the one clean case ("clean") in as a random-audit.
    assert "clean" in {i.case_id for i in a.items}
    assert a.sampled == 1


def test_cases_by_id_enriches_prompt_output():
    rep = _report()
    cases_by_id = {"confident": Case(id="confident", prompt="P?", output="O!")}
    wl = build_worklist(rep, cases_by_id, sample_frac=0.0)
    item = next(i for i in wl.items if i.case_id == "confident")
    assert item.prompt == "P?"
    assert item.output == "O!"
