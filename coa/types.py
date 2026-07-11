"""Data model for cite-or-abstain scoring.

Two orthogonal axes, kept deliberately separate:

  1. *Category* — the stance of the output (cited / uncited-confident /
     uncited-hedged / abstained). Assigned by the LLM categorizer.
  2. *Verification* — whether a `cited` output's citation actually exists in
     the corpus and supports the claim. Established mechanically (passage match)
     plus a narrow LLM call (topical alignment).

A third, non-rubric axis records *disposition* (`CaseStatus`): whether the case
was scorable at all, or whether the scored model emitted an invalid output, or
the harness's own judge failed. Folding these into the four rubric categories
would let a garbage output masquerade as an `abstained` pass, so they are
tracked apart from the rubric.

All pydantic so JSON round-trips are free and validation errors are precise.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    """The four mutually exclusive categories from the cite-or-abstain rubric."""

    CITED = "cited"
    UNCITED_CONFIDENT = "uncited-confident"
    UNCITED_HEDGED = "uncited-hedged"
    ABSTAINED = "abstained"


class CaseStatus(str, Enum):
    """Per-case disposition, orthogonal to the rubric category.

    Kept separate from `Category` so a non-scorable case cannot be silently
    counted as (e.g.) an `abstained` pass.
    """

    SCORED = "scored"
    INVALID_OUTPUT = "invalid-output"  # the *scored model* emitted empty/non-substantive output
    JUDGE_FAILED = "judge-failed"  # the *harness's own* categorizer LLM returned unusable output
    ERROR = "error"  # the harness raised an exception while scoring this case (crash-isolated)


class TopicalAlignment(str, Enum):
    """Does a cited passage actually support the claim it's attached to?"""

    SUPPORTS = "supports"
    UNRELATED = "unrelated"
    CONTRADICTS = "contradicts"
    UNCERTAIN = "uncertain"


class PassageMatch(str, Enum):
    """Could the cited passage/section be located in the corpus?

    Three-valued on purpose. Collapsing "could not check" (no corpus supplied)
    into "not found" would manufacture false hallucinations; collapsing it into
    "found" — the v0.1 behaviour — silently trusts the model's self-reported
    citation. Neither is acceptable, so `UNVERIFIABLE` is a first-class state.
    """

    FOUND = "found"
    NOT_FOUND = "not-found"
    UNVERIFIABLE = "unverifiable"


class CitationVerdict(str, Enum):
    """The derived, single-word verdict for one citation.

    Distinguishes the two failure modes the harness exists to separate:
    a *fabricated* citation (the reference does not exist in the corpus) from a
    *miscited* one (the reference exists but does not support the claim).
    """

    VERIFIED = "verified"  # located in corpus AND passage supports the claim
    MISCITED = "miscited"  # located in corpus BUT passage is unrelated / contradicts
    FABRICATED = "fabricated"  # not locatable in corpus (the classic hallucinated citation)
    UNCERTAIN = "uncertain"  # located, but the judge cannot confirm support
    UNVERIFIABLE = "unverifiable"  # no corpus was available to check against


class Citation(BaseModel):
    """A citation extracted from an LLM output."""

    source: str = Field(
        description="The cited source (e.g., 'Synthetic Prostate Cancer Guideline 2024')."
    )
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
    expected_category: Category | None = None
    expected_correct: bool | None = Field(
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
    parse_ok: bool = Field(
        default=True,
        description="False when the categorizer LLM's own response could not be parsed.",
    )


class VerificationResult(BaseModel):
    """Output of the verifier step for a single citation."""

    citation: Citation
    passage_match: PassageMatch
    topical_alignment: TopicalAlignment
    match_method: str = Field(
        default="",
        description="Matcher tier that located the passage: substring / section-id / token / none.",
    )
    verifier_rationale: str = ""

    @property
    def passage_found(self) -> bool:
        """Back-compatible boolean view: True only when the passage was located."""
        return self.passage_match == PassageMatch.FOUND

    @property
    def verdict(self) -> CitationVerdict:
        """Collapse the two raw signals into one auditable verdict."""
        if self.passage_match == PassageMatch.UNVERIFIABLE:
            return CitationVerdict.UNVERIFIABLE
        if self.passage_match == PassageMatch.NOT_FOUND:
            return CitationVerdict.FABRICATED
        # passage located -> alignment decides
        if self.topical_alignment == TopicalAlignment.SUPPORTS:
            return CitationVerdict.VERIFIED
        if self.topical_alignment in (TopicalAlignment.UNRELATED, TopicalAlignment.CONTRADICTS):
            return CitationVerdict.MISCITED
        return CitationVerdict.UNCERTAIN


class CaseScore(BaseModel):
    """Per-case result."""

    case_id: str
    status: CaseStatus = CaseStatus.SCORED
    category: Category | None = None
    citations: list[Citation] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    rationale: str = ""
    expected_category: Category | None = None
    expected_correct: bool | None = None

    @property
    def correctly_categorized(self) -> bool | None:
        if self.expected_category is None or self.category is None:
            return None
        return self.category == self.expected_category


class RunReport(BaseModel):
    """Aggregate report for a run."""

    model: str
    cases: list[CaseScore]
    counts: dict[str, int]  # rubric-category distribution over scorable cases
    status_counts: dict[str, int]  # disposition distribution over all cases
    verdict_counts: dict[str, int]  # citation-verdict distribution over all verifications
    coverage_cited_correct: float
    coverage_ci: tuple[float, float] = (0.0, 0.0)
    confident_error_rate: float  # penalized cell: uncited-confident-wrong OR fabricated/miscited
    rate_ci: tuple[float, float] = (0.0, 0.0)
    categorizer_accuracy: float | None = None
    judge_failure_rate: float = 0.0  # harness reliability: share of cases the judge couldn't parse
    error_rate: float = 0.0  # harness reliability: share of cases that crashed during scoring
    judge_vs_mechanical_delta: int = 0  # citations a string-match passes but the judge rejects
    scored_denominator: int = 0  # denominator for coverage/rate (excludes judge-failed + errored)
    lambda_: float
    score: float
    frozen_judge: dict[str, str] = Field(default_factory=dict)  # pinned prompt/model fingerprint
    corpus: dict[str, str] = Field(default_factory=dict)  # pinned corpus id/version/sha fingerprint
