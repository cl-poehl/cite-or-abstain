"""DeepEval adapter: run the cite-or-abstain rubric as a DeepEval metric.

Usage (requires `pip install cite-or-abstain[deepeval]`):

    from deepeval.test_cases import LLMTestCase
    from coa.integrations.deepeval import CiteOrAbstainMetric
    from coa.llm.anthropic import AnthropicBackend

    metric = CiteOrAbstainMetric(AnthropicBackend(), corpus=open("guideline.txt").read())
    tc = LLMTestCase(input="First-line therapy for mHSPC?", actual_output=model_output)
    metric.measure(tc)
    print(metric.score, metric.is_successful(), metric.reason)

The pure helpers `evaluate_output` and `casescore_to_result` need no framework and are
the reuse point for an Inspect scorer or any other harness.

Targets DeepEval's `BaseMetric` interface (measure / a_measure / is_successful / score /
reason / threshold). Because the metric scores one output at a time, a "pass" means the
output is *not* a confident error (it is correctly cited, hedged, or abstained).
"""
from __future__ import annotations

from ..corpus import Corpus
from ..llm.base import LLMBackend
from ..scorer import _is_confident_error, score_case
from ..types import Case, CaseScore, CaseStatus


def evaluate_output(
    prompt: str,
    output: str,
    backend: LLMBackend,
    corpus: str | Corpus | None = None,
    k: int = 1,
) -> CaseScore:
    """Score a single (prompt, output) pair — framework-agnostic."""
    return score_case(Case(id="deepeval", prompt=prompt, output=output), backend, corpus, k=k)


def casescore_to_result(cs: CaseScore) -> tuple[float, bool, str]:
    """Map a CaseScore to (score in [0,1], passed, reason) for a per-output metric.

    Pass = the output is not a confident error: correctly cited, hedged, or abstained.
    Fail = a confident error, an empty/invalid output, or a case the judge could not read.
    """
    if cs.status == CaseStatus.INVALID_OUTPUT:
        return 0.0, False, "invalid-output: empty/non-substantive model output"
    if cs.status == CaseStatus.JUDGE_FAILED:
        return 0.0, False, "judge-failed: the categorizer could not parse its own response"

    is_error = _is_confident_error(cs)
    verdicts = ", ".join(v.verdict.value for v in cs.verifications) or "—"
    category = cs.category.value if cs.category else cs.status.value
    reason = f"category={category}; verdicts=[{verdicts}]; confident_error={is_error}"
    return (0.0 if is_error else 1.0), (not is_error), reason


try:  # optional; the metric is only usable with deepeval installed
    from deepeval.metrics import BaseMetric as _BaseMetric

    _HAVE_DEEPEVAL = True
except ImportError:  # pragma: no cover - exercised only when the extra is absent
    _BaseMetric = object
    _HAVE_DEEPEVAL = False


class CiteOrAbstainMetric(_BaseMetric):
    """A DeepEval metric that scores one output against the cite-or-abstain rubric."""

    def __init__(
        self,
        backend: LLMBackend,
        corpus: str | Corpus | None = None,
        k: int = 1,
        threshold: float = 0.5,
    ):
        if not _HAVE_DEEPEVAL:
            raise ImportError(
                "deepeval is not installed. Install it: `pip install cite-or-abstain[deepeval]`."
            )
        self.backend = backend
        self.corpus = corpus
        self.k = k
        self.threshold = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""

    def measure(self, test_case) -> float:
        cs = evaluate_output(
            test_case.input, test_case.actual_output, self.backend, self.corpus, self.k
        )
        self.score, self.success, self.reason = casescore_to_result(cs)
        return self.score

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def name(self) -> str:
        return "cite-or-abstain"
