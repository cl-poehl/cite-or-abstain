"""Adapters that expose the cite-or-abstain rubric inside other eval frameworks.

The rubric earns adoption fastest as a metric inside the harness a team already runs.
`coa.integrations.deepeval` provides a DeepEval `BaseMetric`. The pure mapping helpers
(`evaluate_output`, `casescore_to_result`) are framework-agnostic and importable without
any framework installed, so an Inspect scorer or a custom harness can reuse them.
"""
from .deepeval import casescore_to_result, evaluate_output

__all__ = ["casescore_to_result", "evaluate_output"]
