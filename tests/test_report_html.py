"""HTML report renderer tests — structure + safety, no LLM."""
from coa.report_html import render_report_html
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


def _report():
    scores = [
        CaseScore(
            case_id="a",
            status=CaseStatus.SCORED,
            category=Category.CITED,
            citations=[Citation(source="EAU 2024", section="§6.4.2")],
            verifications=[
                VerificationResult(
                    citation=Citation(source="EAU 2024", section="§6.4.2"),
                    passage_match=PassageMatch.FOUND,
                    topical_alignment=TopicalAlignment.SUPPORTS,
                )
            ],
            expected_category=Category.CITED,
        ),
        CaseScore(case_id="b", status=CaseStatus.JUDGE_FAILED, category=None),
    ]
    return compile_report(
        scores, "test/model", corpus={"corpus_id": "x", "corpus_version": "1", "corpus_sha": "abc"}
    )


def test_html_is_self_contained_and_complete():
    html = render_report_html(_report())
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "<style>" in html  # inline CSS
    # No external asset references.
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html.lower()


def test_html_includes_key_report_fields():
    html = render_report_html(_report())
    assert "test/model" in html
    assert "verified" in html
    assert "judge-failed" in html
    assert "corpus x" in html


def test_html_escapes_untrusted_text():
    scores = [
        CaseScore(case_id="<img src=x onerror=alert(1)>", status=CaseStatus.SCORED,
                  category=Category.ABSTAINED),
    ]
    html = render_report_html(compile_report(scores, "m"))
    assert "<img src=x" not in html
    assert "&lt;img src=x" in html


def test_html_handles_no_corpus():
    html = render_report_html(compile_report([], "m"))
    assert "no corpus" in html
