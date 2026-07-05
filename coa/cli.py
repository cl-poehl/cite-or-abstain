"""coa · command-line interface for cite-or-abstain.

Usage:

    coa score --cases examples/cases.json --corpus examples/synthetic_corpus.txt
    coa score --provider openai --model gpt-4o --cases mycases.json
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .adjudicate import build_worklist
from .backends import CachingBackend, MeteredBackend, RetryingBackend
from .corpus import Corpus
from .llm.anthropic import AnthropicBackend
from .llm.base import LLMBackend
from .llm.openai import OpenAIBackend
from .report_html import render_report_html
from .scorer import compile_report, frozen_judge_fingerprint, score_cases
from .types import Case, CaseStatus, Category, RunReport
from .validation import validate_judge

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


def _make_backend(provider: str, model: str | None) -> LLMBackend:
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
    corpus: Path | None = typer.Option(
        None, "--corpus", help="Plain-text source corpus for citation verification."
    ),
    provider: str = typer.Option(
        "anthropic", "--provider", "-p", help="LLM backend: anthropic or openai."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Specific model id (e.g., claude-sonnet-4-6)."
    ),
    lambda_: float = typer.Option(
        5.0,
        "--lambda",
        "-l",
        help="Penalty weight for uncited-confident-incorrect rate. Default 5.0 (clinical).",
    ),
    k: int = typer.Option(
        1,
        "--k",
        help="Judge draws per case; k>1 takes a majority vote (frozen-judge stabiliser).",
    ),
    concurrency: int = typer.Option(
        1, "--concurrency", "-j", help="Score cases concurrently on N threads (I/O-bound)."
    ),
    retries: int = typer.Option(
        2, "--retries", help="Retry each API call on transient failure (0 to disable)."
    ),
    cache: Path | None = typer.Option(
        None, "--cache", help="JSON cache file: memoize identical calls across runs."
    ),
    price_in: float = typer.Option(
        0.0, "--price-in", help="USD per 1K input tokens, to estimate run cost."
    ),
    price_out: float = typer.Option(
        0.0, "--price-out", help="USD per 1K output tokens, to estimate run cost."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON report to file."
    ),
    html: Path | None = typer.Option(
        None, "--html", help="Write a self-contained HTML report to file."
    ),
):
    """Score a set of LLM outputs against the cite-or-abstain rubric."""

    cases_data = json.loads(cases.read_text(encoding="utf-8"))
    case_list = [Case(**c) for c in cases_data]
    corpus_obj = Corpus.from_path(corpus) if corpus else None

    # Compose resilience/metering wrappers around the raw backend.
    raw = _make_backend(provider, model)
    wrapped: LLMBackend = raw
    if retries > 0:
        wrapped = RetryingBackend(wrapped, retries=retries)
    if cache is not None:
        wrapped = CachingBackend(wrapped, cache)
    metered = MeteredBackend(wrapped)
    backend = metered

    if corpus_obj is None:
        console.print(
            "[yellow]No --corpus supplied: cited passages cannot be located, so every "
            "citation is UNVERIFIABLE and coverage is a lower bound.[/yellow]"
        )
    else:
        m = corpus_obj.manifest
        console.print(f"[dim]corpus {m.id} v{m.version or '—'} · sha {corpus_obj.sha}[/dim]")
    console.print(
        f"\n[cyan]Scoring {len(case_list)} cases · backend [bold]{backend.name}[/bold] · "
        f"λ={lambda_} · k={k} · concurrency={concurrency}[/cyan]\n"
    )

    case_scores = score_cases(case_list, backend, corpus_obj, k=k, max_workers=concurrency)

    frozen = frozen_judge_fingerprint(backend.name, k)
    corpus_fp = corpus_obj.fingerprint() if corpus_obj else {}
    report = compile_report(
        case_scores, backend.name, lambda_=lambda_, frozen_judge=frozen, corpus=corpus_fp
    )

    table = Table(title=f"cite-or-abstain · {backend.name}", show_lines=False)
    table.add_column("case", style="dim", no_wrap=True)
    table.add_column("category", style="bold")
    table.add_column("citations", overflow="fold")
    table.add_column("verification", overflow="fold")
    table.add_column("expected", style="dim")

    for cs in case_scores:
        if cs.status != CaseStatus.SCORED:
            label = cs.status.value
            cat_cell = f"[magenta]{label}[/magenta]"
        else:
            cat_style = _CATEGORY_STYLE.get(cs.category.value, "white")
            cat_cell = f"[{cat_style}]{cs.category.value}[/{cat_style}]"
        cite_str = " · ".join(c.source for c in cs.citations) or "—"
        if cs.verifications:
            verify_str = " · ".join(v.verdict.value for v in cs.verifications)
        else:
            verify_str = "—"
        expected_str = cs.expected_category.value if cs.expected_category else "—"
        cc = cs.correctly_categorized
        match = "✓" if cc else ("✗" if cc is False else "")
        table.add_row(
            cs.case_id,
            cat_cell,
            cite_str,
            verify_str,
            f"{expected_str} {match}".strip(),
        )

    console.print(table)
    console.print()

    counts_str = " · ".join(f"[bold]{k}[/bold]={v}" for k, v in report.counts.items())
    console.print(f"counts        {counts_str}")
    if report.verdict_counts:
        verdict_str = " · ".join(f"[bold]{k}[/bold]={v}" for k, v in report.verdict_counts.items())
        console.print(f"verdicts      {verdict_str}")
    cov_lo, cov_hi = report.coverage_ci
    n_scored = report.scored_denominator
    console.print(
        f"coverage      [bold]{report.coverage_cited_correct:.3f}[/bold]   "
        f"[dim]95% CI [{cov_lo:.3f}, {cov_hi:.3f}][/dim]   (correctly-cited, n={n_scored})"
    )
    rate_lo, rate_hi = report.rate_ci
    console.print(
        f"conf-error    [bold red]{report.confident_error_rate:.3f}[/bold red]   "
        f"[dim]95% CI [{rate_lo:.3f}, {rate_hi:.3f}][/dim]   (confident + unsourced/fabricated)"
    )
    if report.categorizer_accuracy is not None:
        console.print(
            f"categorizer   [bold]{report.categorizer_accuracy:.3f}[/bold]   "
            "(accuracy vs expected_category labels)"
        )
    if report.judge_failure_rate > 0:
        console.print(
            f"judge-failed  [bold yellow]{report.judge_failure_rate:.3f}[/bold yellow]   "
            "(harness reliability — cases the judge could not parse, excluded from denominator)"
        )
    if report.error_rate > 0:
        console.print(
            f"errored       [bold yellow]{report.error_rate:.3f}[/bold yellow]   "
            "(harness crashes, isolated per-case, excluded from denominator)"
        )
    if report.judge_vs_mechanical_delta > 0:
        console.print(
            f"judge>match   [bold]{report.judge_vs_mechanical_delta}[/bold]   "
            "(citations a string-match would accept but the judge flagged as unsupported)"
        )
    console.print(
        f"\n[bold cyan]score = coverage − λ·conf-error "
        f"= {report.coverage_cited_correct:.3f} − {lambda_}·"
        f"{report.confident_error_rate:.3f} "
        f"= {report.score:+.3f}[/bold cyan]\n"
    )

    # Usage / cost line.
    u = metered.usage
    usage_str = f"{u['calls']} calls · {u['input_tokens']}+{u['output_tokens']} tokens"
    if price_in or price_out:
        usage_str += f" · ~${metered.cost(price_in, price_out):.4f}"
    if isinstance(wrapped, CachingBackend):
        usage_str += f" · cache {wrapped.hits} hit / {wrapped.misses} miss"
    console.print(f"[dim]{usage_str}[/dim]")

    if output:
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]report written to {output}[/dim]")
    if html:
        html.write_text(render_report_html(report), encoding="utf-8")
        console.print(f"[dim]HTML report written to {html}[/dim]")


@app.command()
def adjudicate(
    report: Path = typer.Option(
        ..., "--report", "-r", help="A JSON report produced by `coa score -o report.json`."
    ),
    cases: Path | None = typer.Option(
        None, "--cases", "-c", help="Original cases JSON, to add prompt/output to the worklist."
    ),
    sample_frac: float = typer.Option(
        0.1, "--sample", help="Fraction of clean cases to add as a random audit."
    ),
    seed: int = typer.Option(0, "--seed", help="RNG seed for the random audit (reproducible)."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the worklist JSON to file."
    ),
):
    """Emit the human-review worklist from a scored run (offline; no backend needed)."""
    rep = RunReport.model_validate_json(report.read_text(encoding="utf-8"))
    cases_by_id = None
    if cases:
        cases_data = json.loads(cases.read_text(encoding="utf-8"))
        cases_by_id = {c["id"]: Case(**c) for c in cases_data}

    worklist = build_worklist(rep, cases_by_id, sample_frac=sample_frac, seed=seed)

    console.print(
        f"\n[cyan]Adjudication worklist · {len(worklist.items)}/{worklist.total_cases} cases "
        f"({worklist.flagged} flagged · {worklist.sampled} random-audit @ seed {seed})[/cyan]\n"
    )
    table = Table(show_lines=False)
    table.add_column("case", style="dim", no_wrap=True)
    table.add_column("category / status", style="bold")
    table.add_column("verdicts", overflow="fold")
    table.add_column("reasons", overflow="fold")
    for it in worklist.items:
        cat = it.category or it.status
        table.add_row(
            it.case_id,
            cat,
            " · ".join(it.verdicts) or "—",
            " · ".join(it.reasons),
        )
    console.print(table)
    console.print()

    if output:
        output.write_text(worklist.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]worklist written to {output}[/dim]")


@app.command(name="validate-judge")
def validate_judge_cmd(
    report: Path = typer.Option(
        ..., "--report", "-r", help="A JSON report from `coa score -o` over a LABELED set."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the validation JSON to file."
    ),
):
    """Measure the categorizer against human `expected_category` labels.

    The categorizer is itself an LLM judge; this reports whether to trust it. Run it on a
    real, human-labeled, held-out set — not the toy examples.
    """
    rep = RunReport.model_validate_json(report.read_text(encoding="utf-8"))
    v = validate_judge(rep)

    if v.n_labeled == 0:
        console.print(
            "[yellow]No labeled cases found. Add `expected_category` to your cases and "
            "re-run `coa score` before validating the judge.[/yellow]"
        )
        raise typer.Exit(code=1)

    lo, hi = v.accuracy_ci
    console.print(
        f"\n[cyan]Categorizer validation · n={v.n_labeled} labeled cases · "
        f"model {rep.model}[/cyan]\n"
    )
    console.print(
        f"accuracy    [bold]{v.accuracy:.3f}[/bold]   [dim]95% CI [{lo:.3f}, {hi:.3f}][/dim]"
    )
    console.print(
        f"Gwet AC1    [bold]{v.gwet_ac1:.3f}[/bold]   "
        "[dim](skew-robust agreement; prefer over κ)[/dim]"
    )
    if v.n_labeled < 30:
        console.print(
            f"[yellow]n={v.n_labeled} is too small to trust — this is illustrative only. "
            "Validate on a real labeled set (≥ a few dozen cases).[/yellow]"
        )

    # Confusion matrix: rows = expected (human), cols = predicted (categorizer).
    cats = [c.value for c in Category]
    table = Table(title="confusion — rows: expected, cols: predicted", show_lines=False)
    table.add_column("expected ↓ / pred →", style="dim", no_wrap=True)
    for c in cats:
        table.add_column(c, justify="right")
    for exp in cats:
        row = [exp]
        for pred in cats:
            n = v.confusion.get(exp, {}).get(pred, 0)
            cell = str(n) if n else "·"
            row.append(f"[green]{n}[/green]" if (exp == pred and n) else cell)
        table.add_row(*row)
    console.print(table)
    console.print()

    if output:
        output.write_text(v.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]validation written to {output}[/dim]")


@app.command()
def report(
    report: Path = typer.Option(..., "--report", "-r", help="A JSON report from `coa score -o`."),
    output: Path = typer.Option(..., "--output", "-o", help="Write the HTML report here."),
):
    """Render a saved JSON report as a self-contained HTML file."""
    rep = RunReport.model_validate_json(report.read_text(encoding="utf-8"))
    output.write_text(render_report_html(rep), encoding="utf-8")
    console.print(f"[dim]HTML report written to {output}[/dim]")


@app.command()
def version():
    """Print the installed version."""
    console.print(f"cite-or-abstain v{__version__}")
