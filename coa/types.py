"""Data model for cite-or-abstain scoring.

Four mutually exclusive output categories; structured citation; per-case
verification; per-run aggregate report. All pydantic so JSON round-trips
are free and validation errors are precise.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    """The four mutually exclusive categories from the cite-or-abstain rubric."""

    CITED = "cited"
    UNCITED_CONFIDENT = "uncited-confident"
    UNCITED_HEDGED = "uncited-hedged"
    ABSTAINED = "abstained"


class TopicalAlignment(str, Enum):
    """Does a cited passage actually support the claim it's attached to?"""

    SUPPORTS = "supports"
    UNRELATED = "unrelated"
    CONTRADICTS = "contradicts"
    UNCERTAIN = "uncertain"


class Citation(BaseModel):
    """A citation extracted from an LLM output."""

    source: str = Field(description="The cited source (e.g., 'EAU Guidelines on Prostate Cancer 2024').")
    section: str = Field(default="", description="The cited section identifier (e.g., '§6.4.2').")
    passage: str = Field(
        default="",
        description="Verbatim or paraphrased passage the LLM claims is in the source.",
    )


class Case(BaseModel):
    """A single case to score against the cite-or-abstain rubric.

    Either expected_category alone (categorizer-correctness test) or
    expected_category + expected_correct (full pipeline test).
    """

    id: str
    prompt: str = Field(description="The clinical prompt or question.")
    output: str = Field(description="The LLM output to score.")
    expected_category: Optional[Category] = None
    expected_correct: Optional[bool] = Field(
        default=None,
        description=(
            "For 'cited' or 'uncited-confident', whether the underlying medical claim "
            "is actually correct. None means 'unknown' (the scorer treats unknowns as the "
            "conservative default per category)."
        ),
    )


class Categorization(BaseModel):
    """Output of the categorizer step."""

    category: Category
    citations: list[Citation] = Field(default_factory=list)
    rationale: str = ""


class VerificationResult(BaseModel):
    """Output of the verifier step for a single citation."""

    citation: Citation
    passage_found: bool
    topical_alignment: TopicalAlignment
    verifier_rationale: str = ""


class CaseScore(BaseModel):
    """Per-case result."""

    case_id: str
    category: Category
    citations: list[Citation] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    rationale: str = ""
    expected_category: Optional[Category] = None
    expected_correct: Optional[bool] = None

    @property
    def correctly_categorized(self) -> Optional[bool]:
        if self.expected_category is None:
            return None
        return self.category == self.expected_category


class RunReport(BaseModel):
    """Aggregate report for a run."""

    model: str
    cases: list[CaseScore]
    counts: dict[str, int]
    coverage_cited_correct: float
    rate_uncited_confident_incorrect: float
    categorizer_accuracy: Optional[float] = None
    lambda_: float
    score: float
