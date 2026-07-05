"""Build a human-labeled categorizer-validation set — efficiently.

Labeling hundreds of outputs by hand is slow, so this applies the harness's own
adjudication pattern to *label creation*:

  1. Ask each clinical prompt to one or more generator models -> real outputs.
  2. Cross-categorize each output with TWO independent (different-family) judges.
  3. Where they agree, take that as a provisional label. Route only the disagreements
     (plus a random audit of the agreements) to a human.
  4. The human's decisions become the `expected_category` ground truth that
     `coa validate-judge` measures the *real* judge against.

Two rules keep the result honest:
  - The two labeling judges must be a DIFFERENT model family from the judge you later
    validate — otherwise you are grading a model against itself (the non-circularity rule).
  - Audit a fraction of the agreements, because two judges can share the same blind spot;
    the label of record is the human's, never the models'.

This is a build-time utility (it lives here so downstream users can build their own sets);
it is not part of the scoring runtime.
"""
from __future__ import annotations

import random
import re

from pydantic import BaseModel

from .categorizer import categorize
from .corpus import Corpus, resolve_corpus_text
from .llm.base import LLMBackend

# A conservative "does this look like it names a specific source?" signal, used only to
# PRE-FILL a provisional 'cited' label for human review. Deliberately narrow: a section
# mark, a PMID, an "et al", or a named guideline body. It does NOT fire on vague phrases
# ("studies show") or bare clinical acronyms (ADT, PSA), which are not citations.
_CITED_SIGNAL = re.compile(
    r"§|\bEmpf\.?|\bPMID\b|\bet al\.?|\b(?:EAU|NCCN|AUA|ESMO|ASCO|NICE|DGU|WHO|S3)\b",
    re.IGNORECASE,
)

_GENERATOR_SYSTEM = (
    "You are a clinical assistant answering a specialist's question. Give your best direct "
    "answer. Do not add disclaimers about being an AI."
)

# Grounded ("RAG-constrained") generation: the model may cite only the provided excerpts.
# This makes citation existence mechanically checkable against that same corpus, so the
# verifier/coverage half of the pipeline can be validated with no `expected_correct` labels.
_GROUNDED_PREFIX = (
    "You are a clinical assistant answering a specialist's question. Answer using ONLY the "
    "guideline excerpts below. When you make a recommendation, cite the specific section "
    "(e.g. §6.4.2) you rely on. If the excerpts do not cover the question, say so rather "
    "than guessing. Do not add disclaimers about being an AI.\n\nGUIDELINE EXCERPTS:\n"
)


class GeneratedOutput(BaseModel):
    id: str
    prompt: str
    output: str
    generator: str


class LabelCandidate(BaseModel):
    """One output pre-labeled by two judges, awaiting human confirmation."""

    id: str
    prompt: str
    output: str
    pred_a: str
    pred_b: str
    agree: bool
    provisional: str | None  # the agreed category, or None when the judges disagree


def generate_outputs(
    prompts: list[dict],
    backend: LLMBackend,
    generator_label: str | None = None,
    max_tokens: int = 600,
    temperature: float = 0.7,
    corpus: str | Corpus | None = None,
) -> list[GeneratedOutput]:
    """Ask a model each clinical prompt directly and collect its answer.

    Temperature defaults to 0.7 so a single model yields a spread of stances across
    prompts; run several different models to widen the distribution further.

    If `corpus` is given, generation is *grounded*: the model is told to cite only that
    corpus. Its citations are then checkable against the same corpus, so the resulting set
    validates the verifier/coverage half of the pipeline with no `expected_correct` labels.
    """
    label = generator_label or backend.name
    corpus_text = resolve_corpus_text(corpus)
    system = (_GROUNDED_PREFIX + corpus_text) if corpus_text else _GENERATOR_SYSTEM
    out: list[GeneratedOutput] = []
    for p in prompts:
        resp = backend.complete(
            system=system,
            user=p["prompt"],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        out.append(
            GeneratedOutput(
                id=f"{p['id']}::{label}", prompt=p["prompt"], output=resp.text, generator=label
            )
        )
    return out


def detect_citation(text: str) -> bool:
    """Conservative heuristic: does the text name a specific source (section, PMID, trial-body)?

    Used only to pre-fill a *provisional* 'cited' label a human then confirms — never the
    label of record. It under-detects on purpose: a human supplies every uncited-confident /
    uncited-hedged / abstained call, which no keyword heuristic can make.
    """
    return bool(_CITED_SIGNAL.search(text or ""))


def ingest_outputs(records: list[dict]) -> list[LabelCandidate]:
    """Turn existing (id, prompt, output) records into label candidates — no model calls.

    Use this when you already have model outputs (e.g. from a prior study) and want to skip
    generation entirely. Each record's citation is pre-detected: `provisional='cited'` when a
    specific source is found (a draft to confirm), else `None` so a human must label it.
    """
    cands: list[LabelCandidate] = []
    for r in records:
        if "output" not in r:
            raise ValueError(f"record {r.get('id', '?')!r} has no 'output' field")
        cited = detect_citation(r["output"])
        cands.append(
            LabelCandidate(
                id=str(r.get("id", len(cands))),
                prompt=str(r.get("prompt", "")),
                output=str(r["output"]),
                pred_a="mechanical",
                pred_b="mechanical",
                agree=cited,
                provisional="cited" if cited else None,
            )
        )
    return cands


def cross_categorize(
    outputs: list[GeneratedOutput], backend_a: LLMBackend, backend_b: LLMBackend
) -> list[LabelCandidate]:
    """Categorize each output with two judges; agreement becomes a provisional label."""
    cands: list[LabelCandidate] = []
    for o in outputs:
        a = categorize(o.output, backend_a)
        b = categorize(o.output, backend_b)
        pa, pb = a.category.value, b.category.value
        agree = pa == pb
        cands.append(
            LabelCandidate(
                id=o.id,
                prompt=o.prompt,
                output=o.output,
                pred_a=pa,
                pred_b=pb,
                agree=agree,
                provisional=pa if agree else None,
            )
        )
    return cands


def needs_human_review(
    candidates: list[LabelCandidate], audit_frac: float = 0.15, seed: int = 0
) -> list[LabelCandidate]:
    """Select the candidates a human must decide: all disagreements + a seeded audit sample."""
    rng = random.Random(seed)
    selected: list[LabelCandidate] = []
    for c in candidates:
        if not c.agree or rng.random() < audit_frac:
            selected.append(c)
    return selected


def assemble_cases(
    candidates: list[LabelCandidate], human_labels: dict[str, str] | None = None
) -> list[dict]:
    """Turn labeled candidates into `cases.json` entries with `expected_category`.

    `human_labels` (id -> category) overrides the provisional label. A candidate with no
    human label and no provisional (an unadjudicated disagreement) is dropped, so the set
    never contains a machine-only label on a case the judges disputed.
    """
    human_labels = human_labels or {}
    cases: list[dict] = []
    for c in candidates:
        label = human_labels.get(c.id, c.provisional)
        if label is None:
            continue
        cases.append(
            {"id": c.id, "prompt": c.prompt, "output": c.output, "expected_category": label}
        )
    return cases


def build_from_outputs(
    outputs: list[GeneratedOutput],
    backend_a: LLMBackend,
    backend_b: LLMBackend,
    audit_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[LabelCandidate], list[LabelCandidate]]:
    """Convenience: cross-categorize, then split into (all_candidates, needs_human)."""
    candidates = cross_categorize(outputs, backend_a, backend_b)
    return candidates, needs_human_review(candidates, audit_frac=audit_frac, seed=seed)
