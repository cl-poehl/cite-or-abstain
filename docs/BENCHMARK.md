# Example run — what the harness output looks like

**This is not a benchmark or a validation.** It is a small illustrative run — three models,
32 questions, a *synthetic* corpus, a single unvalidated judge — included only to show the
shape of the harness's output and how to read it. The numbers below are **not evidence about
these models**; with this sample size, corpus, and setup they are illustrative at best. Read
the caveats before drawing any conclusion.

`scripts/benchmark.py` runs each model over the same grounded questions and scores every
answer with one fixed judge (so no model grades itself). Example output:

**judge = gpt-4o · corpus = synthetic-demo (synthetic, ~5 sections) · 32 prompts · single run**

| model | cited | uncited-confident | abstained | verified | fabricated | conf-error | score |
|---|---|---|---|---|---|---|---|
| claude-sonnet-5 | 19% | 0% | 78% | 6 | 0 | 0.03 | +0.00 |
| claude-haiku-4-5 | 12% | 3% | 84% | 3 | 3 | 0.09 | −0.41 |
| gpt-4o-mini | 0% | 12% | 81% | 0 | 0 | 0.12 | −0.62 |

How to read a row: the score is `coverage − λ·conf-error`; the verdict counts show whether
cited claims were `verified` (found + supported) or `fabricated` (not in the corpus). *In this
particular run*, the three rows happened to look different — but that difference is not
established, for the reasons below.

## Why you should not read this as a result

- **Tiny sample, single run.** The whole picture rests on single-digit event counts (e.g. 3
  vs 0 `fabricated`) from one stochastic generation. No confidence intervals, no repeats — the
  differences are plausibly noise.
- **Synthetic, incomplete corpus.** The ~80% abstention is a corpus artifact: the small
  synthetic corpus doesn't cover most questions, so grounded models abstain. And a model that
  cites a *real* guideline section absent from this fake corpus is flagged `fabricated` — so
  `fabricated` here conflates "hallucinated" with "cited something we don't have."
- **Unvalidated judge and matcher.** The verdicts come from an unvalidated judge plus a simple
  string matcher; some `fabricated` counts could be matcher misses rather than real
  hallucinations. Validating the judge (`coa validate-judge`) is a prerequisite for trusting
  any of these numbers — see [`VALIDATION.md`](VALIDATION.md).

A trustworthy comparison would need a real corpus, more prompts with reported intervals,
multiple runs, a validated judge, and a manual spot-check of the flagged citations.

## Reproduce

```bash
python scripts/benchmark.py \
  --prompts examples/validation_prompts.json --corpus examples/corpus \
  --judge openai:gpt-4o \
  --models anthropic:claude-sonnet-5 openai:gpt-4o-mini anthropic:claude-haiku-4-5-20251001 \
  --out benchmark.json
```
