"""cite-or-abstain · evaluation harness for clinical LLM outputs.

Categorizes LLM outputs into four mutually exclusive classes (cited,
uncited-confident, uncited-hedged, abstained), verifies cited passages
against a source corpus, and produces a headline metric:

    score = coverage(cited-correct) - lambda * confident_error_rate

Methodology and design rationale: docs/METHODOLOGY.md
"""

__version__ = "0.6.4"

from .adjudicate import AdjudicationItem, Worklist, build_worklist
from .backends import CachingBackend, MeteredBackend, RetryingBackend
from .corpus import Corpus, CorpusManifest
from .report_html import render_report_html
from .scorer import compile_report, frozen_judge_fingerprint, score_case, score_cases
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

__all__ = [
    "AdjudicationItem",
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
    "PassageMatch",
    "RetryingBackend",
    "RunReport",
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
