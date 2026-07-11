"""Smallest possible end-to-end example.

Requires ANTHROPIC_API_KEY in the environment. Run from the project root:

    python examples/basic.py
"""
from __future__ import annotations

import os
from pathlib import Path

from coa import Case, Category
from coa.llm.anthropic import AnthropicBackend
from coa.scorer import compile_report, score_case

if "ANTHROPIC_API_KEY" not in os.environ:
    raise SystemExit("Set ANTHROPIC_API_KEY in your environment to run this example.")

case = Case(
    id="basic-001",
    prompt="What is first-line systemic therapy for mHSPC?",
    output=(
        "For metastatic hormone-sensitive prostate cancer (mHSPC), the recommended "
        "first-line therapy is androgen deprivation therapy combined with either docetaxel "
        "or a novel hormonal agent. Per Synthetic Prostate Cancer Guideline 2024 §6.4.2, "
        "intensified systemic therapy at metastatic diagnosis improves overall survival."
    ),
    expected_category=Category.CITED,
    expected_correct=True,
)

corpus = (Path(__file__).parent / "synthetic_corpus.txt").read_text()
backend = AnthropicBackend()

result = score_case(case, backend, corpus=corpus)
print(f"Case:      {result.case_id}")
print(f"Category:  {result.category.value}")
for c in result.citations:
    print(f"  Citation: {c.source} {c.section}")
for v in result.verifications:
    print(
        f"  Verify:   passage_found={v.passage_found}, "
        f"alignment={v.topical_alignment.value}"
    )

report = compile_report([result], backend.name)
print(f"\nScore: {report.score:+.3f}")
