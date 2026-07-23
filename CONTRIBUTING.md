# Contributing

Thanks for taking a look. This is a small, deliberately opinionated library; the fastest way
to get a change merged is to understand the two principles it is built around.

## The two principles

**1. Never manufacture a verdict the evidence doesn't support.**
Every state that means *"we could not check"* is first-class and must stay distinguishable
from *"we checked and it failed"*. `PassageMatch` is three-valued (`found` / `not-found` /
`unverifiable`) for this reason, and `CitationVerdict` separates `fabricated` (positive
evidence of absence) from `unlocated` (the matcher could not find it — inconclusive).
Collapsing these is the bug class this project exists to avoid: it reports a matcher
limitation as a hallucination rate.

**2. The penalty must not be gameable.**
The scored model must never be able to improve its score by sounding *more* confident or by
attaching a source it made up. Hedged and abstained outputs are never penalised; a bad
citation is penalised rather than scored a free zero. If a change would let an unfindable
made-up citation escape the penalty, it needs a very good argument — see the reasoning on
`PENALIZED_VERDICTS` in `coa/scorer.py`.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -e ".[dev]"
pytest -q
ruff check .
```

`[dev]` pulls in both judge SDKs because the test suite imports them. The library core is
deliberately SDK-free — please keep `anthropic` / `openai` imports out of the core import
path (`coa/llm/__init__.py` lazy-loads them; there is a test that asserts `import coa`
leaves both out of `sys.modules`).

## Tests

- **No test may require a network call or an API key.** Use the fake backends in `tests/`.
- Deterministic tiers (passage matching, scoring, verdict derivation) should be tested
  directly rather than through a judge.
- Two golden tests pin stable output shapes. Regenerate deliberately, never reflexively:
  ```bash
  pytest tests/test_golden_report.py --update-goldens
  ```
  If a golden changes, the diff belongs in the PR description with a sentence on *why* the
  new value is correct.
- When you touch verdict derivation, add a case per *cause*, not per outcome — see
  `tests/test_verifier_verdicts.py::test_not_found_verdict_splits_by_cause`.

## Style

- `ruff check .` must pass; line length is 100.
- Comments explain *why*, not *what*. The non-obvious clinical or statistical reasoning is
  the part worth writing down.
- Public behaviour changes need a `CHANGELOG.md` entry, and anything that moves a reported
  number needs it called out explicitly — including when the default is intentionally left
  unchanged.

## Scope

In scope: the rubric, the verifier, scoring, judge validation, reporting, and backends.

Out of scope: anything that turns this into a general-purpose RAG-evaluation framework. The
value here is a narrow, auditable rubric with a threat model, not breadth. If you want
answer-quality or retrieval metrics, the tools in the README's *Related work* section do that
well and this library is happy to sit alongside them.

## Reporting problems

For a scoring or verdict bug, the most useful report is a minimal `VerificationResult` or
`CaseScore` fixture plus the verdict you expected and why. For anything security-relevant in a
clinical deployment context, please open a private report rather than a public issue.
