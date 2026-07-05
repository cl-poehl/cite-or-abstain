"""Judge-validation harness tests."""
from coa.scorer import compile_report
from coa.types import CaseScore, CaseStatus, Category
from coa.validation import validate_judge


def _labeled(case_id, predicted, expected):
    return CaseScore(
        case_id=case_id,
        status=CaseStatus.SCORED,
        category=predicted,
        expected_category=expected,
    )


def test_no_labels_returns_empty():
    report = compile_report([CaseScore(case_id="a", category=Category.CITED)], "m")
    v = validate_judge(report)
    assert v.n_labeled == 0
    assert v.accuracy is None


def test_accuracy_confusion_and_ac1():
    scores = [
        _labeled("a", Category.CITED, Category.CITED),  # correct
        _labeled("b", Category.UNCITED_CONFIDENT, Category.UNCITED_CONFIDENT),  # correct
        _labeled("c", Category.UNCITED_HEDGED, Category.UNCITED_CONFIDENT),  # wrong
    ]
    report = compile_report(scores, "m")
    v = validate_judge(report)

    assert v.n_labeled == 3
    assert abs(v.accuracy - 2 / 3) < 1e-9
    # confusion[expected][predicted]
    assert v.confusion["cited"]["cited"] == 1
    assert v.confusion["uncited-confident"]["uncited-hedged"] == 1
    assert v.confusion["uncited-confident"]["uncited-confident"] == 1
    assert v.gwet_ac1 is not None
    lo, hi = v.accuracy_ci
    assert lo <= v.accuracy <= hi


def test_judge_failed_cases_excluded_from_validation():
    scores = [
        _labeled("a", Category.CITED, Category.CITED),
        CaseScore(
            case_id="b",
            status=CaseStatus.JUDGE_FAILED,
            category=None,
            expected_category=Category.CITED,
        ),
    ]
    report = compile_report(scores, "m")
    v = validate_judge(report)
    assert v.n_labeled == 1  # the judge-failed case is not counted as a categorization
