#!/usr/bin/env python
"""Build a human-labeled categorizer-validation set from a prompt bank.

Two phases (run generate first, label the worklist by hand, then assemble):

  # 1. Generate outputs from N models, cross-categorize with two OTHER models,
  #    and emit a review worklist (disagreements + audit sample) to label.
  python scripts/build_validation_set.py generate \
      --prompts examples/validation_prompts.json \
      --generators anthropic:claude-sonnet-4-6 openai:gpt-4o openai:gpt-4o-mini \
      --judge-a anthropic:claude-sonnet-4-6 --judge-b openai:gpt-4o \
      --out-candidates candidates.json --out-worklist worklist.json

  #   -> open worklist.json, set "label" on each item (a human decides), save as labels.json
  #      (labels.json = {"<candidate id>": "cited" | "uncited-confident" | ...})

  # 2. Assemble the final validation set from candidates + your human labels.
  python scripts/build_validation_set.py assemble \
      --candidates candidates.json --labels labels.json \
      --out examples/validation_set.json

Then: coa score --cases examples/validation_set.json --corpus <corpus> -o report.json
      coa validate-judge --report report.json

Note the judges (--judge-a/--judge-b) must be a DIFFERENT family from the model you later
validate as the real judge, or you are grading a model against itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coa.corpus import Corpus
from coa.labeling import (
    LabelCandidate,
    assemble_cases,
    build_from_outputs,
    generate_outputs,
    ingest_outputs,
)
from coa.llm.anthropic import AnthropicBackend
from coa.llm.base import LLMBackend
from coa.llm.openai import OpenAIBackend


def _backend(spec: str) -> LLMBackend:
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicBackend(model=model) if model else AnthropicBackend()
    if provider == "openai":
        return OpenAIBackend(model=model) if model else OpenAIBackend()
    raise SystemExit(f"unknown provider in {spec!r} (use anthropic:... or openai:...)")


def cmd_generate(args: argparse.Namespace) -> None:
    prompts = json.loads(Path(args.prompts).read_text("utf-8"))["prompts"]
    corpus = Corpus.from_path(args.corpus) if args.corpus else None
    if corpus is not None:
        print(f"grounded generation on corpus {corpus.manifest.id} (models cite only this)")
    outputs = []
    for spec in args.generators:
        b = _backend(spec)
        print(f"generating {len(prompts)} outputs with {spec} ...")
        outputs.extend(generate_outputs(prompts, b, generator_label=spec, corpus=corpus))

    candidates, worklist = build_from_outputs(
        outputs, _backend(args.judge_a), _backend(args.judge_b),
        audit_frac=args.audit_frac, seed=args.seed,
    )
    Path(args.out_candidates).write_text(
        json.dumps([c.model_dump() for c in candidates], indent=2), "utf-8"
    )
    # Worklist items carry an empty "label" for the human to fill.
    items = [{**c.model_dump(), "label": ""} for c in worklist]
    Path(args.out_worklist).write_text(json.dumps(items, indent=2), "utf-8")

    agree = sum(1 for c in candidates if c.agree)
    print(
        f"{len(candidates)} outputs · {agree} judge-agreements · "
        f"{len(worklist)} to review -> {args.out_worklist}"
    )


def _write_candidates_and_worklist(cands, out_candidates: str, out_worklist: str) -> None:
    Path(out_candidates).write_text(
        json.dumps([c.model_dump() for c in cands], indent=2), "utf-8"
    )
    items = [{**c.model_dump(), "label": c.provisional or ""} for c in cands]
    Path(out_worklist).write_text(json.dumps(items, indent=2), "utf-8")


def cmd_ingest(args: argparse.Namespace) -> None:
    p = Path(args.inputs)
    if p.suffix.lower() == ".csv":
        import csv

        with p.open(encoding="utf-8") as f:
            records = list(csv.DictReader(f))
    else:
        data = json.loads(p.read_text("utf-8"))
        records = data.get("outputs", data.get("cases", data)) if isinstance(data, dict) else data

    cands = ingest_outputs(records)
    _write_candidates_and_worklist(cands, args.out_candidates, args.out_worklist)
    pre = sum(1 for c in cands if c.provisional)
    print(
        f"{len(cands)} outputs · {pre} pre-filled 'cited' (confirm) · "
        f"{len(cands) - pre} to label -> {args.out_worklist}"
    )


def cmd_assemble(args: argparse.Namespace) -> None:
    candidates = [
        LabelCandidate(**c) for c in json.loads(Path(args.candidates).read_text("utf-8"))
    ]
    raw = json.loads(Path(args.labels).read_text("utf-8"))
    # Accept either {"id": "label"} or a labeled worklist [{"id":..., "label":...}].
    if isinstance(raw, list):
        human = {i["id"]: i["label"] for i in raw if i.get("label")}
    else:
        human = {k: v for k, v in raw.items() if v}

    cases = assemble_cases(candidates, human)
    Path(args.out).write_text(json.dumps(cases, indent=2), "utf-8")
    print(f"assembled {len(cases)} labeled cases -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(required=True)

    g = sub.add_parser("generate", help="generate outputs + emit a labeling worklist")
    g.add_argument("--prompts", required=True)
    g.add_argument("--generators", nargs="+", required=True, help="provider:model specs")
    g.add_argument("--judge-a", required=True)
    g.add_argument("--judge-b", required=True)
    g.add_argument(
        "--corpus",
        default=None,
        help="Ground generation on this corpus (dir or .txt) so citations are checkable.",
    )
    g.add_argument("--out-candidates", default="candidates.json")
    g.add_argument("--out-worklist", default="worklist.json")
    g.add_argument("--audit-frac", type=float, default=0.15)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_generate)

    i = sub.add_parser("ingest", help="ingest EXISTING outputs (JSON/CSV) — no model calls")
    i.add_argument("--inputs", required=True, help="JSON list or CSV of {id,prompt,output}")
    i.add_argument("--out-candidates", default="candidates.json")
    i.add_argument("--out-worklist", default="worklist.json")
    i.set_defaults(func=cmd_ingest)

    a = sub.add_parser("assemble", help="assemble the validation set from labels")
    a.add_argument("--candidates", required=True)
    a.add_argument("--labels", required=True)
    a.add_argument("--out", default="examples/validation_set.json")
    a.set_defaults(func=cmd_assemble)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
