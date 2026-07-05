# The evidence behind the design

`cite-or-abstain` is small, but almost every design decision in it is a response
to a specific, published finding about how clinical LLMs fail. This document
records those findings and the design choice each one drove, so the harness reads
as *operationalized evidence* rather than opinion.

Two scope notes:

- These are **external, peer-reviewed findings**. This document is not a validation
  study of this tool — see [`METHODOLOGY.md`](METHODOLOGY.md) for why the harness is a
  screen, not a verdict.
- Every empirical number below is quoted from the cited paper. Full references,
  with DOIs, are at the end.

---

## 1. Why categorical failure modes, not aggregate accuracy

**Finding.** Endpoint accuracy hides the failures that matter. Kenaston et al.
(2026, *npj Digital Medicine*) evaluated LLM reasoning on authentic oncology notes
(breast, pancreatic, prostate) and found GPT-4 produced reasoning errors in **23.1%**
of note interpretations — mostly cognitive-bias patterns (confirmation, anchoring,
omission) — and that these errors were "strongly associated with guideline-discordant
recommendations." Their conclusion is blunt: *"Endpoint accuracy alone may mask
clinically meaningful reasoning failures."*

**Finding.** "Agreement with a reference answer" is itself a shaky gold standard.
Naito et al. (2022, *JAMA Network Open*) had 12 expert molecular tumor boards
independently recommend treatment for 50 simulated cases: mean concordance with the
central consensus was **62%** (95% CI 57–65%), ranging **48–86%** across boards.
Experts routinely disagree, so a metric that rewards a model for matching one
reference conflates "correct" with "agreeable." And where models have been checked
against cancer-treatment guidance directly — Chen et al. (2023, *JAMA Oncology*) — the
chatbot output frequently interleaved guideline-concordant and non-concordant
recommendations within a single answer, so a per-*output* accuracy label is too coarse
to capture what actually went wrong.

**Design choice.** The harness scores four categorical dispositions
(`cited` / `uncited-confident` / `uncited-hedged` / `abstained`) and a per-citation
verdict (`verified` / `miscited` / `fabricated` / `uncertain` / `unverifiable`). It
deliberately does **not** emit a single accuracy figure. The headline metric weights
the failure *mode* that carries clinical cost, not raw agreement.

---

## 2. Why citations are verified for *existence*, mechanically — not graded by the judge

**Finding.** Confident clinical text frequently cites nothing real. van Kessel et al.
(2026, *BMJ Health & Care Informatics*) analysed 12,197 diagnostic LLM outputs and
measured guideline **omissions of up to 97%** (DeepSeek-V3; 46% for GPT-4.1),
**hallucinated guidelines up to 9%**, and citation rates swinging from 0% to 78%
purely with patient sociodemographics — concluding that guideline prediction in LLM
outputs is "a stochastic event." A citation's *presence and confident tone* say
nothing about whether the referenced guidance exists.

**Finding.** Grounding a model in retrieved guideline text is what makes its
citations trustworthy — but grounding constrains, it does not guarantee correctness.
Tung et al. (2025, *JMIR*) built a RAG pipeline constrained to retrieved EAU/AUA
excerpts for PSA-testing decisions: **95.5%** guideline-concordant (210/220) versus
junior clinicians at **62.3%** closed-book and **74.1%** open-book. Retrieval grounds
the citation in a real passage; whether that passage actually *supports* the specific
claim is a separate question.

**Design choice.** Verification is two orthogonal axes, mirroring exactly that
distinction:

1. **Passage match** (`coa.verifier.locate_passage`) — a deterministic check that the
   cited passage/section *exists in the corpus*. Its whole point is to verify
   existence, not shape: a well-formed "§6.999" that appears nowhere in the corpus is
   the classic hallucinated citation, and shape-validation sails straight past it.
   This axis is three-valued — `found` / `not-found` / `unverifiable` — because
   collapsing "no corpus to check against" into either "found" (a silent pass, the
   v0.1 bug) or "not-found" (a manufactured hallucination) is wrong. No corpus →
   `unverifiable`, and an unverifiable citation is never counted as correct.
2. **Topical alignment** — a narrow LLM call, consulted *only once a passage is
   located*, deciding whether it supports the claim.

The two axes let the harness name the two failure modes separately: **fabricated**
(the reference does not exist) versus **miscited** (it exists but does not support
the claim).

---

## 3. Why the LLM judge is a screen, not the verdict

**Finding.** LLM evaluators detect that something is wrong but cannot reliably say
*what*. Kenaston et al. (2026) again: automated LLM-based evaluators "detected error
presence but failed to reliably classify subtypes," and a self-mitigation strategy
"yielded only modest improvement." A separate methods literature documents systematic
LLM-judge biases — self-preference, verbosity/length, and position effects — that make
a single judge call an unreliable adjudicator on its own.

**Design choice.**

- The judge is scoped as narrowly as possible: it makes the *topical-alignment*
  call and the *categorization* call, and nothing else. Citation existence — the part
  a judge is worst at and a string matcher is best at — is handled mechanically.
- A judge parse failure is a **recorded outcome** (`judge-failed`), reported as a
  harness-reliability rate and *excluded from the scored denominator*, never silently
  relabelled as a real category.
- A **frozen-judge** mode pins the model id, temperature, prompt version + SHA, and an
  optional k-sample majority vote into every run report, so a run is re-scorable.
  (Multi-draw self-consistency is *opt-in*: it is not assumed to help a categorical
  judgment and should be measured before it is trusted.)
- Per the screen-not-verdict pattern ([`METHODOLOGY.md`](METHODOLOGY.md)): auto-score, then
  route the `uncertain`, `judge-failed`, and a random subset to human adjudication. This
  harness produces the screen; it does not replace the adjudicator.

---

## 4. Why the deterministic tiers are the auditable core

**Finding.** Structure beats free generation, by a wide margin, on protocol adherence.
Arriola-Montenegro et al. (2025, *Frontiers in AI*) compared a loosely-prompted LLM
against deterministic prompt logic for hemodialysis anemia dosing: **32%** protocol
adherence (loose) versus **100%** (deterministic, 300/300), eliminating unsafe and
mistimed recommendations.

**Finding.** An LLM navigating a guideline decision tree is not yet reliable as a
standalone CDSS. Delourme et al. (2025, *Methods Inf Med*) coupled LLMs to the
OncoDoc2 breast-cancer guideline system: despite **75.6%** question-answering
accuracy, only **16.67%** of the resulting recommendations matched the gold-standard
CDSS output, because "any deviation from a criterion alters the recommendations
generated."

**Design choice.** The harness leans on its deterministic tiers (passage match,
section-id existence, structural checks) as the reproducible, auditable core, and
frames the LLM tier (topical alignment, categorization) as the component that is
*not* bit-reproducible — temperature 0 does not make an LLM deterministic. That is
why the LLM tier is pinned rather than trusted, and why the score is designed to be
reconstructable from the recorded verdicts without re-calling the model.

---

## 5. Why λ penalizes one cell — and never punishes hedging or abstention

**Finding.** The dangerous mode is the *confident, unsourced, wrong* claim. van
Kessel's stochasticity result plus Kenaston's finding that confirmation/anchoring/
omission biases were "most strongly linked to potentially harmful outputs" both point
the same way: a fluent recommendation asserted without a verifiable source, that
happens to be wrong, is the highest-cost failure — and it is invisible to a coverage
metric.

**Design choice.** The metric

```
score = coverage(cited-correct) − λ · rate(uncited-confident-incorrect)
```

applies its penalty to exactly one cell — **uncited, confident, and incorrect** — and
to no other. Hedged and abstained outputs are never penalized. This is deliberate: a
model that hedges or abstains is signalling the uncertainty the evidence says is often
warranted (Tung's grounding gap, Naito's expert disagreement), and a harness that
punished caution would train models to sound confident. λ defaults to `5.0` for
clinical use — one confidently-wrong recommendation outweighs ~five correctly-cited
ones — and should be lowered for general-knowledge domains.

---

## 6. Why every rate is reported with an interval

**Finding / method.** A run of 8–50 cases is a *sample*; a bare point estimate
over-reads noise as signal. The harness reports a Wilson score interval on coverage
and failure rate (Newcombe RG, *Statistics in Medicine* 1998;17:857–872), preferred
over the Wald interval because it stays sensible at the extremes that matter — a
zero-count safety event still yields a non-trivial upper bound.

When comparing the harness's categorical labels against human labels, prefer a
prevalence-robust agreement coefficient (e.g. Gwet's AC1) over a bare Cohen/Fleiss κ:
the four categories are skewed (most safe outputs cluster in one or two classes), and
under skew κ collapses toward zero even at high raw agreement — the well-documented
"kappa paradox" (Feinstein & Cicchetti, *J Clin Epidemiol* 1990). Report a band, not
a single coefficient.

---

## What this is not

This harness is a **first-pass screen**. It does not replace expert adjudication for
the cases the LLM tier gets wrong, and none of the findings above are claims about
*this tool's* validated accuracy — they are the external evidence that shaped its
design. Use it to regression-test model releases on the cite-or-abstain axis, to
compare model behaviour across versions, and to produce audit-ready scoring tables
that a clinician can then adjudicate.

---

## References

All verified against PubMed.

1. van Kessel R, Anderson M, McMillan B, et al. Omission and hallucination prevalence
   of clinical guidelines in diagnostic large language model outputs. *BMJ Health Care
   Inform* 2026;33(1):e101959. https://doi.org/10.1136/bmjhci-2025-101959
2. Kenaston MW, Ayub U, Parmar M, et al. Structured reasoning failures compromise LLM
   interpretation of clinical oncology notes. *NPJ Digit Med* 2026.
   https://doi.org/10.1038/s41746-026-02951-5
3. Tung JYM, Le Q, Yao J, et al. Performance of retrieval-augmented generation large
   language models in guideline-concordant prostate-specific antigen testing:
   comparative study with junior clinicians. *J Med Internet Res* 2025;27:e78393.
   https://doi.org/10.2196/78393
4. Arriola-Montenegro J, Thongprayoon C, Bizer B, et al. A deterministic large
   language model framework for safe, protocol-adherent clinical decision support:
   application in hemodialysis anemia management (AnemiaCare HDs). *Front Artif Intell*
   2025;8:1728320. https://doi.org/10.3389/frai.2025.1728320
5. Delourme S, Redjdal A, Bouaud J, Seroussi B. Leveraging guideline-based clinical
   decision support systems with large language models: a case study with breast
   cancer. *Methods Inf Med* 2025;63(3-04):85–96. https://doi.org/10.1055/a-2528-4299
6. Naito Y, Sunami K, Kage H, et al. Concordance between recommendations from
   multidisciplinary molecular tumor boards and central consensus for cancer treatment
   in Japan. *JAMA Netw Open* 2022;5(12):e2245081.
   https://doi.org/10.1001/jamanetworkopen.2022.45081
7. Chen S, Kann BH, Foote MB, et al. Use of artificial intelligence chatbots for cancer
   treatment information. *JAMA Oncol* 2023;9(10):1459–1462.
   https://doi.org/10.1001/jamaoncol.2023.2954

Statistical-methods references: Newcombe RG. Two-sided confidence intervals for the
single proportion. *Stat Med* 1998;17:857–872. · Feinstein AR, Cicchetti DV. High
agreement but low kappa. *J Clin Epidemiol* 1990;43:543–549.
