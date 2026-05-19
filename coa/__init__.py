"""cite-or-abstain · evaluation harness for clinical LLM outputs.

Categorizes LLM outputs into four mutually exclusive classes (cited,
uncited-confident, uncited-hedged, abstained), verifies cited passages
against a source corpus, and produces a headline metric:

    score = coverage(cited-correct) - lambda * rate(uncited-confident-incorrect)

Methodology: https://carlpoehl.com/writing/cite-or-abstain
"""

__version__ = "0.1.0"

from .scorer import compile_report, score_case
from .types import (
    Case,
    CaseScore,
    Categorization,
    Category,
    Citation,
    RunReport,
    TopicalAlignment,
    VerificationResult,
)

__all__ = [
    "Case",
    "CaseScore",
    "Categorization",
    "Category",
    "Citation",
    "RunReport",
    "TopicalAlignment",
    "VerificationResult",
    "compile_report",
    "score_case",
]
