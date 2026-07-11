"""cite-or-abstain · evaluation harness for clinical LLM outputs.

Categorizes LLM outputs into four mutually exclusive classes (cited,
uncited-confident, uncited-hedged, abstained), verifies cited passages
against a source corpus, and produces a headline metric:

    score = coverage(cited-correct) - lambda * confident_error_rate

Methodology and design rationale: docs/METHODOLOGY.md
"""

__version__ = "0.7.0"

from typing import TYPE_CHECKING

from .adjudicate import AdjudicationItem, Worklist, build_worklist
from .backends import CachingBackend, MeteredBackend, RetryingBackend
from .corpus import Corpus, CorpusManifest
from .report_html import render_report_html
from .scorer import compile_report, frozen_judge_fingerprint, score_case, score_cases
from .semantic import SemanticMatcher
from .types import (
    Case,
    CaseScore,
    CaseStatus,
    Categorization,
    Category,
    Citation,
    CitationVerdict,
    PassageMatch,
    RunReport,
    TopicalAlignment,
    VerificationResult,
)
from .validation import JudgeValidation, validate_judge

if TYPE_CHECKING:  # type-checkers/IDEs see the lazily-loaded backends without importing SDKs
    from .llm import AnthropicBackend, OpenAIBackend

__all__ = [
    "AdjudicationItem",
    "AnthropicBackend",
    "CachingBackend",
    "Case",
    "CaseScore",
    "CaseStatus",
    "Categorization",
    "Category",
    "Citation",
    "CitationVerdict",
    "Corpus",
    "CorpusManifest",
    "JudgeValidation",
    "MeteredBackend",
    "OpenAIBackend",
    "PassageMatch",
    "RetryingBackend",
    "RunReport",
    "SemanticMatcher",
    "TopicalAlignment",
    "VerificationResult",
    "Worklist",
    "build_worklist",
    "compile_report",
    "frozen_judge_fingerprint",
    "render_report_html",
    "score_case",
    "score_cases",
    "validate_judge",
]


def __getattr__(name: str) -> object:
    # AnthropicBackend / OpenAIBackend are re-exported lazily so `import coa` never pulls a
    # vendor SDK; the SDK loads (or raises a clear install hint) only on first access.
    if name in ("AnthropicBackend", "OpenAIBackend"):
        from . import llm

        return getattr(llm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
