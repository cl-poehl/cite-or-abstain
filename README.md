# cite-or-abstain

> An evaluation harness for cite-or-abstain compliance in clinical LLM outputs.

[![CI](https://github.com/cl-poehl/cite-or-abstain/actions/workflows/ci.yml/badge.svg)](https://github.com/cl-poehl/cite-or-abstain/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cl-poehl/cite-or-abstain/blob/main/examples/demo.ipynb)

**[▶ Try it in Colab](https://colab.research.google.com/github/cl-poehl/cite-or-abstain/blob/main/examples/demo.ipynb)** — a 2-minute in-browser walkthrough on the bundled adversarial cases (bring your own API key).

`coa` categorizes an LLM output into one of four mutually exclusive classes, verifies any cited passages against a source corpus, and scores the run against a headline metric tuned to the cost of failure in the target domain.

```
$ coa score --cases examples/cases.json --corpus examples/synthetic_corpus.txt

Scoring 8 cases · backend anthropic/claude-sonnet-4-6 · λ=5.0 · k=1

┌───────────────────────────────┬──────────────────────┬──────────────┬───────────────┐
│ case                          │ category             │ citations    │ verdict       │
├───────────────────────────────┼──────────────────────┼──────────────┼───────────────┤
│ ex-001-cited                  │ cited                │ EAU 2024     │ verified      │
│ ex-002-uncited-confident      │ uncited-confident    │ —            │ —             │
│ ex-003-uncited-hedged         │ uncited-hedged       │ —            │ —             │
│ ex-004-abstained              │ abstained            │ —            │ —             │
│ ex-005-hallucinated-citation  │ cited                │ NCCN v9.7    │ fabricated    │
│ ex-006-cited-correct          │ cited                │ EAU 2024     │ verified      │
│ ex-007-vague-reference        │ uncited-confident    │ —            │ —             │
│ ex-008-abstain-with-request   │ abstained            │ —            │ —             │
└───────────────────────────────┴──────────────────────┴──────────────┴───────────────┘

verdicts      verified=2 · fabricated=1
counts        cited=4 · uncited-confident=2 · uncited-hedged=1 · abstained=1
coverage      0.250   95% CI [0.071, 0.591]   (correctly-cited, n=8)
conf-error    0.375   95% CI [0.152, 0.665]   (confident + unsourced/fabricated)
categorizer   —       (run `coa validate-judge` on a labeled set — see below)

score = coverage − λ·conf-error = 0.250 − 5·0.375 = −1.625
```

The `fabricated` verdict on `ex-005` is the point: the citation is well-formed and
confidently stated, but the passage does not exist in the corpus. Shape-validation
would pass it; existence-verification catches it — and a fabricated citation lands in
the penalized `conf-error` cell, so dressing a confident claim in a fake citation is
never rewarded over stating it plainly.

## Why this matters

Clinical LLMs fail in ways an accuracy score can't see. Across 12,197 diagnostic LLM outputs, models omitted up to **97%** of the relevant clinical-guideline content and **hallucinated** guidelines in up to **9%** of cases — with citation behaviour swinging on patient demographics alone ([van Kessel et al., *BMJ Health & Care Informatics* 2026](https://doi.org/10.1136/bmjhci-2025-101959)). A confident, unsourced, wrong recommendation reads exactly like a correct one, so it survives any aggregate-accuracy metric — which is why aggregate accuracy is a vanity metric in clinical AI.

The failures that matter — confident-uncited claims, miscited references, fabricated citations — are categorical, not continuous, and require categorical scoring. This library implements the **cite-or-abstain rubric** — an actionable clinical claim should carry a verifiable citation or explicitly abstain — as a reusable, model-agnostic harness that flags the confident-but-unsourced claim and the fabricated citation (the outputs that hurt someone) and rewards a model only for citing a verifiable source or explicitly abstaining.

Almost every design choice here — verifying citation *existence* mechanically, scoping the LLM judge narrowly, penalizing only the confident-error cell, reporting intervals rather than point estimates — traces to a specific published finding about how clinical LLMs fail. The thesis is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md); the evidence, with numbers and DOIs, in [`docs/FINDINGS.md`](docs/FINDINGS.md).

## The four categories

| Category              | Meaning                                                                          | Treatment |
|-----------------------|----------------------------------------------------------------------------------|-----------|
| `cited`               | Clinically-actionable claim made with a verifiable, on-topic citation.           | Pass (subject to verifier). |
| `uncited-confident`   | Clinically-actionable claim made in confident voice with no specific source.     | **Failure** (most dangerous). |
| `uncited-hedged`      | Clinically-actionable claim made with hedging language and no specific source.   | Lesser failure. |
| `abstained`           | Model explicitly refused, surfaced a reason, requested input.                    | Pass. |

The categorizer is strict: a vague reference (*"studies show", "the literature supports"*) is not a citation; it is uncited.

## The metric

```
score = coverage(cited-correct) − λ · confident_error_rate
```

- `coverage(cited-correct)` — fraction of cases that are correctly cited.
- `confident_error_rate` — fraction of cases that are a **confident error**: a confident, actionable claim with a demonstrated failure. That is any of:
  - `uncited-confident` and incorrect (or unverifiable, so treated as failure);
  - `cited` with a **fabricated or miscited** citation — a confident claim in false authority;
  - `cited` but the underlying claim is known wrong.
- `λ` — penalty weight. Default `5.0` (clinical-cost setting); set lower for general-knowledge domains.

The default is deliberately punitive on the failure mode that matters most. Designers who maximize *coverage* without weighting by failure cost end up shipping confident wrong answers and calling it *"98% accurate."* The penalty lands on **only** the confident-error cell — hedged and abstained outputs are never penalized, so the harness never trains a model to sound confident to dodge the score. Crucially, a **fabricated citation is penalized, not scored a free zero** — otherwise a model could launder a confident claim by attaching a fake source, which is *more* dangerous, not less.

Two honesty rules keep a run from flattering itself, and every rate ships with a Wilson interval (a run of 8–50 cases is a *sample*):

- A **`judge-failed`** case — the harness's *own* categorizer LLM returned unparseable output — is tooling failure, not the scored model's. It is excluded from the coverage/rate denominator and surfaced separately as `judge_failure_rate`, a reliability number to watch.
- An **`invalid-output`** case — the *scored model* emitted empty/non-substantive output — stays in the denominator (it lowers coverage) and is never counted as an `abstained` pass.

See [`docs/FINDINGS.md`](docs/FINDINGS.md) §5–6 for the evidence behind the single-cell penalty and the intervals.

## Evaluation methodology

Built for *quantified, honest* eval — not a headline number:

- every rate ships with a **Wilson confidence interval** (a 30-case run is a *sample*, not a truth);
- judge–human agreement is measured with **Gwet's AC1**, robust to the class skew where Cohen/Fleiss κ collapses;
- **harness failures** (`judge-failed`, `error`) are held *out* of the model's score, and malformed model output is never counted as an `abstained` pass — so a run cannot flatter itself;
- the LLM judge is **itself measured** against human labels (`coa validate-judge`, reporting accuracy + CI + AC1 + a confusion matrix) — the tool validates its own judge.

**Status, stated plainly.** Run end-to-end against Anthropic and OpenAI backends on a curated adversarial suite (`examples/cases.json`) and a small *illustrative* cross-model comparison ([`docs/BENCHMARK.md`](docs/BENCHMARK.md)). It is **not** yet validated on a real, human-labeled clinical set — that requires your own corpus and labels, and the ~30-minute protocol is in [`docs/VALIDATION.md`](docs/VALIDATION.md). No aggregate accuracy figure is claimed, by design: a number from toy data would be worse than none.

## Install

Not yet on PyPI — install from GitHub:

```bash
pip install "git+https://github.com/cl-poehl/cite-or-abstain"
```

Optional extras — `[fuzzy]` (rapidfuzz, higher-recall passage matching) and `[deepeval]`
(run the rubric as a DeepEval metric):

```bash
pip install "cite-or-abstain[fuzzy] @ git+https://github.com/cl-poehl/cite-or-abstain"
```

Or clone and install editable (recommended for development):

```bash
git clone https://github.com/cl-poehl/cite-or-abstain
cd cite-or-abstain
pip install -e ".[dev]"
```

Provide an API key for the LLM backend you want to use as the judge:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

## Quick start

The bundled example runs eight adversarial cases against a small synthetic corpus:

```bash
coa score --cases examples/cases.json --corpus examples/synthetic_corpus.txt
```

Switch backend:

```bash
coa score --provider openai --model gpt-4o --cases examples/cases.json
```

Programmatically:

```python
from coa import Case, Category
from coa.llm.anthropic import AnthropicBackend
from coa.scorer import compile_report, score_case

case = Case(
    id="demo-001",
    prompt="What is first-line therapy for mHSPC?",
    output="For mHSPC, ADT plus docetaxel or a novel hormonal agent is the standard, per EAU 2024 §6.4.2.",
    expected_category=Category.CITED,
    expected_correct=True,
)

backend = AnthropicBackend()
result = score_case(case, backend, corpus=open("guideline.txt").read())
print(result.category.value)              # "cited"
print(result.verifications[0].topical_alignment.value)  # "supports"

report = compile_report([result], backend.name)
print(report.score)                       # 1.0
```

## How verification works

For `cited` outputs, the verifier runs two orthogonal checks per citation, and
collapses them into one auditable **verdict**:

1. **Passage match** (`coa.verifier.locate_passage`) — does the cited passage actually
   *exist where the citation says it does*? A deterministic, **location-aware** match
   that reports *which* tier fired:
   - a cited **section that isn't in the corpus** → `not-found` (`section-absent`): a
     fabricated reference, even if similar prose appears elsewhere;
   - a cited section that *is* present → the passage must appear **within that section's
     window** (bounded at the next section header); a real passage attributed to the
     **wrong section** → `not-found` (`section-mismatch`) — a class of miscitation a
     whole-corpus bag-of-words match is blind to;
   - no section → the passage is matched against the whole corpus (substring, then fuzzy).

   Fuzzy matching uses `rapidfuzz` when installed (`pip install cite-or-abstain[fuzzy]`)
   and falls back to stdlib `difflib`. The result is three-valued — `found` / `not-found`
   / `unverifiable` — because "no corpus to check against" is neither a pass nor a
   fabrication. **With no `--corpus`, every citation is `unverifiable` and coverage is a
   lower bound.** For a large corpus, swap in a vector matcher by replacing
   `locate_passage`.

2. **Topical alignment** — does the located passage actually *support* the claim? A
   narrow LLM-as-judge call returning `supports / unrelated / contradicts / uncertain`,
   consulted only once a passage is found (a fabricated citation's verdict is already
   settled, so no judge call is spent on it).

The two axes name the two failure modes separately:

| Verdict | Meaning |
|---|---|
| `verified` | Located in corpus **and** the passage supports the claim. |
| `miscited` | Located in corpus **but** the passage is unrelated or contradicts. |
| `fabricated` | Not locatable in the corpus — the classic hallucinated citation. |
| `uncertain` | Located, but the judge cannot confirm support. |
| `unverifiable` | No corpus was available to check against. |

Only `verified` counts toward coverage. For a large corpus, swap in vector similarity
against an embedded corpus by replacing `coa.verifier.locate_passage` (or the
`passage_in_corpus` convenience wrapper).

## Pluggable LLM backends

Anthropic and OpenAI ship bundled. A new backend is a 20-line subclass:

```python
from coa.llm.base import LLMBackend, LLMResponse

class LocalvLLMBackend(LLMBackend):
    @property
    def name(self) -> str:
        return "local/qwen-2.5-72b"

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> LLMResponse:
        # ... your vLLM / Ollama / TGI client here
        return LLMResponse(text=text, model="qwen-2.5-72b")
```

The categorizer and verifier are backend-agnostic — they only require `complete()`.

## Adversarial cases

`examples/cases.json` includes eight cases that probe every category, including:

- A **hallucinated-citation** case (`ex-005`) — the citation is plausible but does not exist in the corpus. The verifier should flag it via passage match.
- A **vague-reference** case (`ex-007`) — *"studies show…"* without a specific source. The categorizer should mark this `uncited-confident`, not `cited`.
- An **abstention-with-request** case (`ex-008`) — model declines and requests specific inputs. The categorizer should mark this `abstained`.

The corpus (`examples/synthetic_corpus.txt`) is **synthetic and not for clinical use** — provided so the verifier can run end-to-end without distributing copyrighted guidelines.

## Corpus identity

A verification result only means something if you know which corpus it was scored
against. Pass a plain `.txt` file and it gets a content-hash identity; pass a directory
with a `manifest.json` and the pinned `(id, version, source_document)` plus content hash
is stamped into the run report:

```
examples/corpus/
  manifest.json   {"id": "synthetic-demo", "version": "2024", "source_document": "corpus.txt"}
  corpus.txt
```

```bash
coa score --cases examples/cases.json --corpus examples/corpus
```

## Adjudication

The harness is a **screen, not a verdict**. `coa adjudicate` turns a saved report into a
human-review worklist: every case the automation is least trustworthy on — `uncited-confident`,
any non-`verified` citation, `judge-failed` / `invalid-output` / `error` — plus a seeded
random audit sample of the clean cases, so systematic categorizer/verifier errors surface
even when nothing tripped a flag.

```bash
coa score --cases examples/cases.json --corpus examples/corpus -o report.json
coa adjudicate --report report.json --cases examples/cases.json --sample 0.1 -o worklist.json
```

This implements the *screen, not a verdict* pattern ([`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)): auto-score, then adjudicate the uncertain, the flagged, and a random subset.

## At scale

Scoring a large case set is I/O-bound on API calls. The runner parallelises across cases,
retries transient failures, memoises identical calls, and reports token usage / cost:

```bash
coa score --cases cases.json --corpus corpus \
  --concurrency 8 \        # score 8 cases at once
  --retries 3 \            # retry transient API failures (so a 429 doesn't drop a case)
  --cache .coa-cache.json \# re-runs after a one-line change don't re-pay
  --price-in 3 --price-out 15   # estimate $ from per-1K-token prices
```

The resilience/metering wrappers are composable backends (`RetryingBackend`,
`CachingBackend`, `MeteredBackend`) and are thread-safe, so they work the same way from
the API:

```python
from coa.backends import MeteredBackend, CachingBackend, RetryingBackend
from coa.llm.anthropic import AnthropicBackend
from coa.scorer import score_cases

backend = MeteredBackend(CachingBackend(RetryingBackend(AnthropicBackend()), "cache.json"))
scores = score_cases(cases, backend, corpus, max_workers=8)
print(backend.usage, backend.cost(price_in_per_1k=3.0, price_out_per_1k=15.0))
```

## HTML report

Produce a self-contained, audit-ready HTML page (inline CSS, no JS, light/dark aware) that
a reviewer can open and read:

```bash
coa score --cases cases.json --corpus corpus --html report.html   # during a run
coa report --report report.json -o report.html                    # from a saved JSON report
```

## Use inside DeepEval

The rubric is also a DeepEval metric, so it drops into a harness you already run:

```python
from deepeval.test_cases import LLMTestCase
from coa.integrations.deepeval import CiteOrAbstainMetric
from coa.llm.anthropic import AnthropicBackend

metric = CiteOrAbstainMetric(AnthropicBackend(), corpus=open("guideline.txt").read())
metric.measure(LLMTestCase(input="First-line therapy for mHSPC?", actual_output=model_output))
print(metric.score, metric.is_successful(), metric.reason)
```

A "pass" means the output is *not* a confident error — it is correctly cited, hedged, or
abstained. The framework-agnostic helpers `evaluate_output` / `casescore_to_result` (in
`coa.integrations`) are the reuse point for an Inspect scorer or a custom harness.

## Validating the judge

The categorizer is itself an LLM judge — so, per its own methodology, **you must measure
it before trusting it.** `coa validate-judge` scores the categorizer against human
`expected_category` labels and reports accuracy (with a Wilson interval), **Gwet's AC1**
(a skew-robust agreement coefficient — preferred over Cohen/Fleiss κ, which collapses
under the class imbalance clinical labels always have), and a confusion matrix showing
*where* it errs:

```bash
coa score --cases labeled_cases.json --corpus your_corpus -o report.json
coa validate-judge --report report.json
```

This is deliberately not pre-baked with a flattering headline number: a high score on a
handful of toy cases is meaningless, so it warns below ~30 labeled cases. To earn trust
in the judge, run it on a real, human-labeled, held-out set and report the band.

**[`docs/VALIDATION.md`](docs/VALIDATION.md) is a plain-language, step-by-step guide** (~30
min of hands-on time). If you already have model outputs, ingest them and skip generation
entirely:

```bash
python scripts/build_validation_set.py ingest --inputs myoutputs.json \
  --out-candidates candidates.json --out-worklist worklist.json
# label worklist.json, then assemble + score + validate-judge (see the guide)
```

## What this is — and isn't

**Is.** A categorical evaluation harness for clinical LLM outputs that implements one specific published methodology. Useful for regression-testing model releases on the cite-or-abstain axis, comparing model behavior across versions, and producing audit-ready scoring tables.

**Isn't.** A finished clinical-validation framework. It does not replace expert adjudication for the cases the LLM judge gets wrong. It is a *first pass* — see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the screen-not-verdict pattern (auto-score → adjudicate uncertain + flagged + random subset).

## Status

**v0.6.x** — building the validation set. `coa.labeling` + `scripts/build_validation_set.py`
turn model outputs into a human-labeled categorizer-validation set: **ingest** outputs you
already have (JSON/CSV, `cited` pre-filled) or **generate** fresh from the bundled
GU-oncology prompt bank, have two independent judges pre-label, and route only their
disagreements + an audit sample to a human. **Grounded generation** (`--corpus`) makes the
verifier/coverage half checkable with no extra labeling.
[`docs/VALIDATION.md`](docs/VALIDATION.md) is a plain-language, step-by-step guide.

**v0.5.0** — production surface: concurrency, resilience, cost, an HTML report, and a
DeepEval metric.

- **Scale & resilience**: `--concurrency` (thread-pool scoring), `--retries` (backoff),
  `--cache` (memoised calls), token-usage + cost reporting — composable, thread-safe
  `RetryingBackend` / `CachingBackend` / `MeteredBackend`.
- **Self-contained HTML report** (`coa report`, `--html`): inline CSS, light/dark, escaped.
- **DeepEval metric** (`coa.integrations.deepeval.CiteOrAbstainMetric`) + framework-agnostic
  mapping helpers.

**v0.4.0** — location-aware verification, a penalty that matches the threat model, and a
judge-validation harness. New since v0.1.0:

- **Location-aware, existence-verifying verifier**: three-valued passage match with a
  `verified` / `miscited` / `fabricated` verdict; catches `section-absent` and
  `section-mismatch` (real passage, wrong section); `rapidfuzz`-or-`difflib` fuzzy match.
- **Threat-model-aligned metric**: the penalty (`confident_error_rate`) covers
  uncited-confident-wrong **and** fabricated/miscited citations, so laundering a confident
  claim with a fake citation is never rewarded. Hedged/abstained are never penalized.
- **`coa validate-judge`**: measure the categorizer against human labels — accuracy + Wilson
  CI + **Gwet AC1** + confusion matrix (the tool practices what it preaches).
- **Honest denominators**: `judge-failed` / `error` (harness) excluded; `invalid-output`
  (model) kept; never an `abstained` pass. Wilson intervals on every rate.
- **`coa adjudicate`**: human-review worklist (flagged + seeded random audit).
- **Corpus identity**: pin `(id, version, source_document)` + content hash into every run.
- **Judge-vs-mechanical delta**, **crash-isolating runner**, **frozen-judge** metadata,
  hardened judge parsing.
- [`docs/FINDINGS.md`](docs/FINDINGS.md) + [`docs/VALIDATION.md`](docs/VALIDATION.md): the
  evidence base and the judge-validation protocol.

Roadmap:

- Vector-similarity passage match for large corpora.
- An [Inspect](https://inspect.aisi.org.uk/) scorer (the DeepEval metric ships now; the
  framework-agnostic mapping helpers are the reuse point).
- Per-category release-blocking thresholds in CI.
- A published, human-labeled categorizer-validation set (see [`docs/VALIDATION.md`](docs/VALIDATION.md)).

Contributions, bug reports, and methodology pushback all welcome.

## Citation

```bibtex
@software{poehl2026citeorabstain,
  author = {Pöhl, Carl Luis},
  title  = {{cite-or-abstain}: an evaluation harness for clinical LLM outputs},
  year   = {2026},
  url    = {https://github.com/cl-poehl/cite-or-abstain},
}
```

## Design & rationale

Everything the library rests on is documented in-repo:

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — the cite-or-abstain thesis and the design principles: why categorical scoring beats aggregate accuracy, the failure-mode taxonomy behind the four categories, "the eval is the system", and "a screen, not a verdict".
- [`docs/FINDINGS.md`](docs/FINDINGS.md) — the peer-reviewed clinical-LLM evidence behind each design decision (van Kessel 2026, Kenaston 2026, Tung 2025, …).
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — a step-by-step guide to validating the categorizer against human labels on your own data.

## License

MIT. See [LICENSE](LICENSE).

---

Carl Luis Pöhl · [github.com/cl-poehl](https://github.com/cl-poehl) · [book a call](https://calendly.com/clpoehl/30min)
