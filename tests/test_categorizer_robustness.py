"""Categorizer + pipeline robustness tests, driven by a scripted fake backend."""
from coa.categorizer import _extract_json, categorize
from coa.corpus import Corpus
from coa.scorer import score_case
from coa.testing import RoutedBackend, ScriptedBackend
from coa.types import Case, CaseStatus, Category


def test_extract_json_strips_think_tags_and_fences():
    raw = "<think>let me reason about this</think>\n```json\n{\"category\": \"abstained\"}\n```"
    assert _extract_json(raw) == '{"category": "abstained"}'


def test_extract_json_ignores_trailing_prose():
    raw = '{"category": "cited", "citations": []} \n\nHope that helps!'
    extracted = _extract_json(raw)
    assert extracted == '{"category": "cited", "citations": []}'


def test_extract_json_returns_none_when_absent():
    assert _extract_json("I refuse to answer in JSON.") is None


def test_categorize_handles_think_wrapped_json():
    backend = ScriptedBackend(['<think>hmm</think>{"category": "abstained", "rationale": "r"}'])
    result = categorize("some output", backend)
    assert result.category == Category.ABSTAINED
    assert result.parse_ok is True


def test_categorize_flags_unparseable_as_parse_error():
    backend = ScriptedBackend(["totally not json"])
    result = categorize("some output", backend)
    assert result.parse_ok is False


def test_score_case_empty_output_is_invalid_without_calling_judge():
    backend = ScriptedBackend(['{"category": "abstained"}'])
    cs = score_case(Case(id="x", prompt="p", output="   "), backend)
    assert cs.status == CaseStatus.INVALID_OUTPUT
    assert backend.calls == 0  # pre-flight short-circuit, no judge call spent


def test_score_case_judge_parse_failure_is_judge_failed_status():
    backend = ScriptedBackend(["not json at all"])
    cs = score_case(Case(id="x", prompt="p", output="a real clinical answer"), backend)
    assert cs.status == CaseStatus.JUDGE_FAILED
    assert cs.category is None


def test_alignment_judges_the_model_output_not_the_prompt():
    """The alignment judge must receive the model's assertion (its output), not the question.

    Otherwise it is asked whether a passage supports a *question*, and returns uncertain.
    """
    seen = {}

    cited = '{"category":"cited","citations":[{"source":"X","section":"§1","passage":"alpha"}]}'

    def route(system, user):
        if "OUTPUT TO CATEGORIZE" in user:
            return cited
        seen["align_user"] = user  # the alignment call
        return "supports"

    corpus = Corpus.from_text("§1 alpha beta gamma delta")
    score_case(
        Case(id="c", prompt="QUESTION_TOKEN", output="ANSWER_TOKEN alpha"),
        RoutedBackend(route),
        corpus,
    )
    assert "ANSWER_TOKEN" in seen["align_user"]  # the assertion is the claim
    assert "QUESTION_TOKEN" not in seen["align_user"]  # not the question


def test_k_majority_vote_takes_the_plurality():
    # Two 'abstained' vs one 'cited' -> abstained wins.
    backend = ScriptedBackend(
        [
            '{"category": "abstained"}',
            '{"category": "cited", "citations": [{"source": "X"}]}',
            '{"category": "abstained"}',
        ]
    )
    result = categorize("out", backend, k=3)
    assert result.category == Category.ABSTAINED
    assert backend.calls == 3
