#!/usr/bin/env python
"""Cross-model cite-or-abstain benchmark.

Ask each model the same clinical questions (grounded on a corpus so citations are
mechanically checkable), then score every model's answers with ONE fixed judge — so no
model grades itself. Produces a comparison table: how each model behaves on the
cite-or-abstain axis (does it cite and verify, hedge, abstain, or make confident
unsourced / fabricated claims?).

    python scripts/benchmark.py \
        --prompts examples/validation_prompts.json --corpus examples/corpus \
        --judge openai:gpt-4o \
        --models anthropic:claude-sonnet-5 openai:gpt-4o-mini anthropic:claude-haiku-4-5-20251001 \
        --out benchmark.json

No human labels are needed: the score rewards cited-and-verified answers and penalizes
confident-unsourced or fabricated ones, all determinable against the corpus. It is a
reproducible demonstration on the given corpus, not a clinical validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coa.corpus import Corpus
from coa.labeling import generate_outputs
from coa.llm.anthropic import AnthropicBackend
from coa.llm.base import LLMBackend
from coa.llm.openai import OpenAIBackend
from coa.scorer import compile_report, score_cases
from coa.types import Case


def _backend(spec: str) -> LLMBackend:
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicBackend(model=model) if model else AnthropicBackend()
    if provider == "openai":
        return OpenAIBackend(model=model) if model else OpenAIBackend()
    raise SystemExit(f"unknown provider in {spec!r}")


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "—"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--judge", required=True, help="provider:model used to score every output")
    ap.add_argument("--models", nargs="+", required=True, help="provider:model rows to benchmark")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--lambda", dest="lam", type=float, default=5.0)
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text("utf-8"))["prompts"]
    corpus = Corpus.from_path(args.corpus)
    judge = _backend(args.judge)

    rows = []
    for spec in args.models:
        print(f"[{spec}] generating {len(prompts)} answers ...", flush=True)
        outs = generate_outputs(
            prompts, _backend(spec), generator_label=spec, corpus=corpus,
            max_workers=args.concurrency,
        )
        cases = [Case(id=o.id, prompt=o.prompt, output=o.output) for o in outs]
        print(f"[{spec}] scoring with judge {args.judge} ...", flush=True)
        scores = score_cases(cases, judge, corpus, max_workers=args.concurrency)
        rep = compile_report(scores, spec, lambda_=args.lam, corpus=corpus.fingerprint())
        rows.append(rep)

    # Markdown table.
    cols = ["model", "n", "cited", "unc-conf", "hedged", "abstain",
            "verified", "fabricated", "conf-err", "score"]
    hdr = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    lines = [hdr, sep]
    for r in rows:
        n = r.scored_denominator
        c = r.counts
        v = r.verdict_counts
        lines.append(
            f"| {r.model} | {n} | {_pct(c.get('cited',0),n)} | "
            f"{_pct(c.get('uncited-confident',0),n)} | {_pct(c.get('uncited-hedged',0),n)} | "
            f"{_pct(c.get('abstained',0),n)} | {v.get('verified',0)} | {v.get('fabricated',0)} | "
            f"{r.confident_error_rate:.2f} | {r.score:+.2f} |"
        )
    table = "\n".join(lines)
    print("\njudge = " + args.judge + " · corpus = " + corpus.manifest.id + "\n")
    print(table)

    Path(args.out).write_text(
        json.dumps(
            {"judge": args.judge, "corpus": corpus.fingerprint(), "table_md": table,
             "reports": [r.model_dump(mode="json") for r in rows]},
            indent=2,
        ),
        "utf-8",
    )
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
