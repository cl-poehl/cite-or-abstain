"""Render a RunReport as a single self-contained HTML file.

An "audit-ready scoring table" a clinician or reviewer can actually read: no external
assets (inline CSS, no JS), light/dark aware, everything escaped. Deterministic — no
timestamps or randomness — so the output is stable and testable.
"""
from __future__ import annotations

from html import escape

from .types import CaseStatus, RunReport

_CSS = """
:root {
  color-scheme: light dark;
  --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e3e3e3; --tile:#f7f7f8;
  --green:#127a3d; --red:#b3261e; --amber:#8a6d00; --accent:#2d4b8e;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#161616; --fg:#eaeaea; --muted:#9a9a9a; --line:#333; --tile:#1f1f1f;
    --green:#4cc38a; --red:#ff6b60; --amber:#e0b341; --accent:#8aa6e6;
  }
}
* { box-sizing: border-box; }
body {
  margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
.wrap { max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
.score { font-size: 40px; font-weight: 700; letter-spacing:-.02em; }
.score .formula { font-size: 14px; font-weight: 400; color: var(--muted); }
.tiles {
  display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
  gap:12px; margin:24px 0;
}
.tile { background: var(--tile); border:1px solid var(--line); border-radius:10px; padding:14px; }
.tile .k {
  font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em;
}
.tile .v { font-size:26px; font-weight:600; margin-top:4px; }
.tile .ci { font-size:12px; color:var(--muted); }
h2 {
  font-size:15px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
  border-bottom:1px solid var(--line); padding-bottom:6px; margin:32px 0 12px;
}
.bars { display:flex; flex-direction:column; gap:6px; }
.bar {
  display:grid; grid-template-columns:150px 1fr 42px; align-items:center;
  gap:10px; font-size:13px;
}
.bar .track { background:var(--line); border-radius:5px; height:14px; overflow:hidden; }
.bar .fill { height:100%; background:var(--accent); }
.tblwrap { overflow-x:auto; }
table { border-collapse: collapse; width:100%; font-size:13px; }
th,td {
  text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top;
}
th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; }
td.mono, .fp { font-family: ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.cited{color:var(--green)} .verified{color:var(--green)}
.uncited-confident,.fabricated,.miscited{color:var(--red)}
.uncited-hedged,.uncertain{color:var(--amber)}
.abstained{color:var(--accent)}
.invalid-output,.judge-failed,.error,.unverifiable{color:var(--muted)}
.fp {
  color:var(--muted); background:var(--tile); border:1px solid var(--line);
  border-radius:8px; padding:10px 12px; margin-top:8px; word-break:break-all;
}
"""


def _tile(k: str, v: str, ci: str = "") -> str:
    ci_html = f'<div class="ci">{escape(ci)}</div>' if ci else ""
    return (
        f'<div class="tile"><div class="k">{escape(k)}</div>'
        f'<div class="v">{escape(v)}</div>{ci_html}</div>'
    )


def _bars(title: str, counts: dict[str, int]) -> str:
    if not counts:
        return ""
    total = max(1, sum(counts.values()))
    rows = ""
    for label, n in counts.items():
        pct = 100 * n / total
        cls = escape(label)
        rows += (
            f'<div class="bar"><div class="{cls}">{escape(label)}</div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%"></div></div>'
            f"<div>{n}</div></div>"
        )
    return f"<h2>{escape(title)}</h2><div class='bars'>{rows}</div>"


def _ci(pair) -> str:
    return f"95% CI [{pair[0]:.3f}, {pair[1]:.3f}]"


def _case_rows(report: RunReport) -> str:
    rows = ""
    for cs in report.cases:
        if cs.status != CaseStatus.SCORED:
            cat = cs.status.value
        else:
            cat = cs.category.value if cs.category else "—"
        cites = escape(" · ".join(c.source for c in cs.citations) or "—")
        verdicts = " · ".join(
            f'<span class="{escape(v.verdict.value)}">{escape(v.verdict.value)}</span>'
            for v in cs.verifications
        ) or "—"
        exp = cs.expected_category.value if cs.expected_category else "—"
        mark = ""
        if cs.correctly_categorized is True:
            mark = " ✓"
        elif cs.correctly_categorized is False:
            mark = " ✗"
        rows += (
            f"<tr><td class='mono'>{escape(cs.case_id)}</td>"
            f'<td><span class="{escape(cat)}">{escape(cat)}</span></td>'
            f"<td>{cites}</td><td>{verdicts}</td>"
            f"<td>{escape(exp)}{mark}</td></tr>"
        )
    return rows


def render_report_html(report: RunReport, title: str = "cite-or-abstain report") -> str:
    """Return a complete, self-contained HTML document for a run report."""
    corpus = report.corpus
    corpus_line = (
        f"corpus {corpus.get('corpus_id','?')} v{corpus.get('corpus_version') or '—'} "
        f"· sha {corpus.get('corpus_sha','?')}"
        if corpus
        else "no corpus (citations unverifiable; coverage is a lower bound)"
    )
    cat_acc = (
        f"{report.categorizer_accuracy:.3f}"
        if report.categorizer_accuracy is not None
        else "— (run validate-judge)"
    )

    tiles = "".join(
        [
            _tile("coverage", f"{report.coverage_cited_correct:.3f}", _ci(report.coverage_ci)),
            _tile("conf-error", f"{report.confident_error_rate:.3f}", _ci(report.rate_ci)),
            _tile("categorizer acc", cat_acc),
            _tile("judge-failed", f"{report.judge_failure_rate:.3f}"),
            _tile("errored", f"{report.error_rate:.3f}"),
            _tile("judge>match", str(report.judge_vs_mechanical_delta)),
        ]
    )

    fp = report.frozen_judge
    fp_line = " · ".join(f"{escape(k)}={escape(str(v))}" for k, v in fp.items()) if fp else "—"

    lam = report.lambda_
    formula = (
        f"score = coverage − λ·conf-error = {report.coverage_cited_correct:.3f} "
        f"− {lam}·{report.confident_error_rate:.3f}"
    )

    sub = (
        f"model <b>{escape(report.model)}</b> · λ={lam} · "
        f"n={report.scored_denominator} scored · {escape(corpus_line)}"
    )
    thead = (
        "<tr><th>case</th><th>category / status</th><th>citations</th>"
        "<th>verdicts</th><th>expected</th></tr>"
    )
    body = f"""
    <div class="wrap">
      <h1>{escape(title)}</h1>
      <div class="sub">{sub}</div>
      <div class="score">{report.score:+.3f}<div class="formula">{escape(formula)}</div></div>
      <div class="tiles">{tiles}</div>
      {_bars("categories", report.counts)}
      {_bars("citation verdicts", report.verdict_counts)}
      {_bars("dispositions", report.status_counts)}
      <h2>cases</h2>
      <div class="tblwrap"><table>
        <thead>{thead}</thead>
        <tbody>{_case_rows(report)}</tbody>
      </table></div>
      <h2>frozen judge</h2>
      <div class="fp">{fp_line}</div>
    </div>
    """
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
