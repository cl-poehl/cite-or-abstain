# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] — 2026-07-23

### Added
- **`unlocated` citation verdict.** An unmatched citation is now split by *cause* rather than
  collapsed into `fabricated`:
  | verdict | cause | evidence of fabrication? |
  |---|---|---|
  | `fabricated` | cited section id is absent from the corpus | yes — positive evidence of absence |
  | `miscited` | section real but passage attributed elsewhere, or judge says it doesn't support | yes — wrong attribution |
  | `unlocated` | no section cited and the matcher could not find the passage | **no** — inconclusive |
- `scorer.PENALIZED_VERDICTS` / `PENALIZED_VERDICTS_STRICT_EVIDENCE` — the penalty surface as
  one auditable constant.
- `compile_report(..., penalize_unlocated=False)` — opt out of penalizing `unlocated` once a
  paraphrase-capable matcher (`coa.semantic.SemanticMatcher`) makes that signal trustworthy.
- `coa/py.typed` — the package now advertises its inline type hints to downstream type checkers.
- Tests covering each `NOT_FOUND` cause independently; the report golden pins all four verdicts.

### Changed
- **Reporting only — the default headline score is unchanged.** `unlocated` remains penalized
  by default: a lexical matcher cannot distinguish a grounded paraphrase it missed from a
  passage the model invented, so exempting it would reopen the citation-laundering incentive
  the harness exists to close. Accuracy is recovered through *visibility* (the split is now in
  `verdict_counts`) and through a stronger matcher, not by weakening the penalty.
- A wrong-section citation (`section-mismatch`) now reports as `miscited` rather than
  `fabricated`, matching what it actually is. Same penalty, more accurate label.
- `examples/corpus/corpus.txt` is now byte-identical to `examples/synthetic_corpus.txt`,
  removing a silent divergence between the two bundled example corpora. Verified not to change
  any bundled verdict (passage matching is deterministic).

### Why this matters
A high `confident_error_rate` was previously ambiguous: it could mean the model hallucinated
citations, *or* that the lexical matcher failed on grounded paraphrases (content-token overlap
can approach 1.0 while lexical similarity sits near 0.6). Those demand opposite responses. The
verdicts are now counted separately, so a rate driven by `fabricated` is a real hallucination
finding, while one driven by `unlocated` is a prompt to use a better matcher.

## [0.7.0]

### Added
- `coa.semantic.SemanticMatcher` — embedding-based passage matching, a drop-in `locate=`
  replacement that recovers grounded paraphrases the lexical matcher misses.
- Open-weight / on-prem judge support: any OpenAI-compatible server via `base_url`, plus
  reasoning-trace and harmony-channel parsing.
- Bundled synthetic judge-validation set (36 labelled cases) and `examples/report.json`, so
  `coa report` / `coa adjudicate` run with no API key.
- Unicode-aware (NFC-normalising) passage matching.

### Changed
- **SDK-free core.** `anthropic` and `openai` moved out of the base dependencies into optional
  extras (`[anthropic]`, `[openai]`, `[all]`), so a library-only or on-prem install stays light.
  `import coa` no longer pays SDK import cost.
- README reframed the bundled synthetic set as a *functional demonstration*, not a validation
  result — a perfect score on a small author-labelled set against a 13-section corpus is
  expected and uninformative.

## [0.6.x]
- Validation-set tooling: `coa.labeling` + `scripts/build_validation_set.py` turn model outputs
  into a human-labelled categorizer-validation set (ingest existing outputs or generate fresh).

## [0.5.0]
- Production surface: concurrency, retry/metering backend wrappers, cost accounting, HTML
  report, and adjudication worklists.

## [0.4.0]
- Location-aware verification (a passage must appear *within* the cited section's window), a
  penalty aligned to the threat model, and the three-valued `PassageMatch`.
