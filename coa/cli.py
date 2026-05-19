"""coa · command-line interface for cite-or-abstain.

Usage:

    coa score --cases examples/cases.json --corpus examples/synthetic_corpus.txt
    coa score --provider openai --model gpt-4o --cases mycases.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .llm.anthropic import AnthropicBackend
from .llm.base import LLMBackend
from .llm.openai import OpenAIBackend
from .scorer import compile_report, score_case
from .types import Case

app = typer.Typer(
    help="cite-or-abstain · an evaluation harness for clinical LLM outputs.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

_CATEGORY_STYLE = {
    "cited": "green",
    "uncited-confident": "red",
    "uncited-hedged": "yellow",
    "abstained": "cyan",
}


def _make_backend(provider: str, model: Optional[str]) -> LLMBackend:
    p = provider.lower().strip()
    if p == "anthropic":
        return AnthropicBackend(model=model) if model else AnthropicBackend()
    if p == "openai":
        return OpenAIBackend(model=model) if model else OpenAIBackend()
    raise typer.BadParameter(
        f"Unknown provider: {provider!r}. Use 'anthropic' or 'openai'."
    )


@app.command()
def score(
    cases: Path = typer.Option(..., "--cases", "-c", help="JSON file: list of cases."),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus", help="Plain-text source corpus for citation verification."
    ),
    provider: str = typer.Option(
        "anthropic", "--provider", "-p", help="LLM backend: anthropic or openai."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Specific model id (e.g., claude-sonnet-4-6)."
    ),
    lambda_: float = typer.Option(
        5.0,
        "--lambda",
        "-l",
        help="Penalty weight for uncited-confident-incorrect rate. Default 5.0 (clinical).",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write JSON report to file."
    ),
):
    """Score a set of LLM outputs against the cite-or-abstain rubric."""

    cases_data = json.loads(cases.read_text(encoding="utf-8"))
    case_list = [Case(**c) for c in cases_data]
    corpus_text = corpus.read_text(encoding="utf-8") if corpus else None
    backend = _make_backend(provider, model)

    console.print(
        f"\n[cyan]Scoring {len(case_list)} cases · backend [bold]{backend.name}[/bold] · "
        f"λ={lambda_}[/cyan]\n"
    )

    case_scores = []
    for case in case_list:
        cs = score_case(case, backend, corpus_text)
        case_scores.append(cs)

    report = compile_report(case_scores, backend.name, lambda_=lambda_)

    table = Table(title=f"cite-or-abstain · {backend.name}", show_lines=False)
    table.add_column("case", style="dim", no_wrap=True)
    table.add_column("category", style="bold")
    table.add_column("citations", overflow="fold")
    table.add_column("verification", overflow="fold")
    table.add_column("expected", style="dim")

    for cs in case_scores:
        cat_style = _CATEGORY_STYLE.get(cs.category.value, "white")
        cite_str = " · ".join(c.source for c in cs.citations) or "—"
        if cs.verifications:
            verify_str = " · ".join(
                f"{'✓' if v.passage_found else '✗'} {v.topical_alignment.value}"
                for v in cs.verifications
            )
        else:
            verify_str = "—"
        expected_str = cs.expected_category.value if cs.expected_category else "—"
        match = "✓" if cs.correctly_categorized else ("✗" if cs.expected_category else "")
        table.add_row(
            cs.case_id,
            f"[{cat_style}]{cs.category.value}[/{cat_style}]",
            cite_str,
            verify_str,
            f"{expected_str} {match}".strip(),
        )

    console.print(table)
    console.print()

    counts_str = " · ".join(f"[bold]{k}[/bold]={v}" for k, v in report.counts.items())
    console.print(f"counts        {counts_str}")
    console.print(
        f"coverage      [bold]{report.coverage_cited_correct:.3f}[/bold]   "
        "(rate of correctly-cited claims)"
    )
    console.print(
        f"failure rate  [bold red]{report.rate_uncited_confident_incorrect:.3f}[/bold red]   "
        "(rate of uncited-confident claims treated as incorrect)"
    )
    if report.categorizer_accuracy is not None:
        console.print(
            f"categorizer   [bold]{report.categorizer_accuracy:.3f}[/bold]   "
            "(accuracy vs expected_category labels)"
        )
    console.print(
        f"\n[bold cyan]score = coverage − λ·rate "
        f"= {report.coverage_cited_correct:.3f} − {lambda_}·{report.rate_uncited_confident_incorrect:.3f} "
        f"= {report.score:+.3f}[/bold cyan]\n"
    )

    if output:
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]report written to {output}[/dim]")


@app.command()
def version():
    """Print the installed version."""
    console.print(f"cite-or-abstain v{__version__}")
