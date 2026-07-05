"""Backend wrapper tests: retry, cache, meter, and concurrent scoring."""
import pytest

from coa.backends import CachingBackend, MeteredBackend, RetryingBackend
from coa.llm.base import LLMBackend, LLMResponse
from coa.llm.openai import _downgrade_params
from coa.scorer import score_cases
from coa.testing import RoutedBackend
from coa.types import Case, CaseStatus


class FlakyBackend(LLMBackend):
    """Fails the first `fail_times` calls, then returns a fixed response."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake/flaky"

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        return LLMResponse(text="ok", model="fake", input_tokens=10, output_tokens=3)


def test_retry_recovers_from_transient_failures():
    inner = FlakyBackend(fail_times=2)
    b = RetryingBackend(inner, retries=3, sleep=lambda _d: None)
    resp = b.complete("s", "u")
    assert resp.text == "ok"
    assert inner.calls == 3  # 2 failures + 1 success
    assert b.attempts == 2  # two backoff sleeps


def test_retry_reraises_after_exhausting():
    inner = FlakyBackend(fail_times=99)
    b = RetryingBackend(inner, retries=2, sleep=lambda _d: None)
    with pytest.raises(RuntimeError):
        b.complete("s", "u")
    assert inner.calls == 3  # initial + 2 retries


def test_cache_hits_second_identical_call(tmp_path):
    inner = FlakyBackend(fail_times=0)
    cache = tmp_path / "cache.json"
    b = CachingBackend(inner, cache)
    r1 = b.complete("sys", "user")
    r2 = b.complete("sys", "user")
    assert r1.text == r2.text == "ok"
    assert inner.calls == 1  # second call served from cache
    assert b.hits == 1 and b.misses == 1
    assert cache.exists()


def test_cache_misses_on_different_prompt(tmp_path):
    inner = FlakyBackend(fail_times=0)
    b = CachingBackend(inner, tmp_path / "c.json")
    b.complete("sys", "user-a")
    b.complete("sys", "user-b")
    assert inner.calls == 2


def test_cache_persists_across_instances(tmp_path):
    cache = tmp_path / "c.json"
    inner1 = FlakyBackend(fail_times=0)
    CachingBackend(inner1, cache).complete("s", "u")
    inner2 = FlakyBackend(fail_times=0)
    b2 = CachingBackend(inner2, cache)
    b2.complete("s", "u")
    assert inner2.calls == 0  # served from the persisted cache
    assert b2.hits == 1


def test_meter_accumulates_tokens_and_cost():
    b = MeteredBackend(FlakyBackend(fail_times=0))
    b.complete("s", "u")
    b.complete("s", "u")
    assert b.usage == {"calls": 2, "input_tokens": 20, "output_tokens": 6}
    # cost = 20/1000*3 + 6/1000*15 = 0.06 + 0.09 = 0.15
    assert abs(b.cost(3.0, 15.0) - 0.15) < 1e-9


def _categorizing_backend():
    def route(system, user):
        return '{"category":"abstained","rationale":"r"}' if "OUTPUT TO CATEGORIZE" in user else "x"
    return RoutedBackend(route)


def test_concurrent_scoring_preserves_order_and_isolates_crashes():
    def route(system, user):
        if "BOOM" in user:
            raise RuntimeError("boom")
        return '{"category":"abstained","rationale":"r"}'

    backend = RoutedBackend(route)
    cases = [
        Case(id=f"c{i}", prompt="q", output=("BOOM" if i == 5 else f"answer {i}"))
        for i in range(12)
    ]
    results = score_cases(cases, backend, max_workers=4)
    assert [r.case_id for r in results] == [f"c{i}" for i in range(12)]  # order preserved
    assert results[5].status == CaseStatus.ERROR
    assert all(r.status == CaseStatus.SCORED for i, r in enumerate(results) if i != 5)


def test_concurrent_matches_serial():
    backend = _categorizing_backend()
    cases = [Case(id=f"c{i}", prompt="q", output=f"answer {i}") for i in range(8)]
    serial = score_cases(cases, backend, max_workers=1)
    concurrent = score_cases(cases, _categorizing_backend(), max_workers=4)
    assert [r.category for r in serial] == [r.category for r in concurrent]


def test_openai_downgrade_drops_temperature():
    kw = {"model": "x", "temperature": 0.0, "max_tokens": 100}
    assert _downgrade_params("temperature is deprecated for this model", kw) is True
    assert "temperature" not in kw and kw["max_tokens"] == 100


def test_openai_downgrade_renames_max_tokens():
    kw = {"model": "x", "max_tokens": 100}
    msg = "unsupported parameter: 'max_tokens'. use 'max_completion_tokens' instead."
    assert _downgrade_params(msg, kw) is True
    assert kw.get("max_completion_tokens") == 100 and "max_tokens" not in kw


def test_openai_downgrade_handles_both_at_once():
    kw = {"model": "x", "temperature": 0.0, "max_tokens": 100}
    assert _downgrade_params("temperature unsupported; use max_completion_tokens", kw) is True
    assert "temperature" not in kw and kw.get("max_completion_tokens") == 100


def test_openai_downgrade_no_change_on_unrelated_error():
    kw = {"model": "x", "temperature": 0.0, "max_tokens": 100}
    assert _downgrade_params("incorrect api key provided", kw) is False
    assert kw == {"model": "x", "temperature": 0.0, "max_tokens": 100}
