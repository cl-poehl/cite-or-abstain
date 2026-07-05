"""Composable backend wrappers: resilience, caching, and metering.

Each wraps any `LLMBackend` and is itself an `LLMBackend`, so they stack in the
spirit of the pluggable-backend design:

    backend = MeteredBackend(CachingBackend(RetryingBackend(AnthropicBackend()), "cache.json"))

- `RetryingBackend` — retry transient failures with exponential backoff, so a
  rate-limit blip doesn't silently drop a case from the denominator.
- `CachingBackend` — memoize identical (name, system, user, params) calls to a JSON
  file, so re-scoring a case set after a one-line change doesn't re-pay for every call.
- `MeteredBackend` — accumulate call/token counts (thread-safe) and compute cost.

All are thread-safe, so they compose with concurrent scoring (`score_cases(max_workers=N)`).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .llm.base import LLMBackend, LLMResponse


class RetryingBackend(LLMBackend):
    """Retry `complete()` on any exception with capped exponential backoff.

    Retrying reduces spurious `error`/`judge-failed` cases from transient API faults
    (429s, timeouts). A call that still fails after all retries re-raises, so a
    genuinely broken case is still isolated as an error by the runner.
    """

    def __init__(
        self,
        inner: LLMBackend,
        retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._inner = inner
        self._retries = retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self.attempts = 0  # total retry sleeps performed (observability)

    @property
    def name(self) -> str:
        return self._inner.name

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> LLMResponse:
        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                return self._inner.complete(system, user, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001 — retry any provider error
                last_exc = e
                if attempt == self._retries:
                    break
                self.attempts += 1
                self._sleep(min(self._max_delay, self._base_delay * (2**attempt)))
        assert last_exc is not None
        raise last_exc


class CachingBackend(LLMBackend):
    """Memoize completions to a JSON file, keyed by (name, system, user, params).

    Deterministic re-runs (same prompts, same judge) become free. The cache is
    content-addressed, so a prompt or model change misses cleanly rather than
    returning a stale answer.
    """

    def __init__(self, inner: LLMBackend, path: str | Path):
        self._inner = inner
        self._path = Path(path)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self._cache: dict[str, dict] = {}
        if self._path.exists():
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))

    @property
    def name(self) -> str:
        return self._inner.name

    def _key(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        blob = "\x00".join(
            [self._inner.name, system, user, str(max_tokens), str(temperature)]
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> LLMResponse:
        key = self._key(system, user, max_tokens, temperature)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            with self._lock:
                self.hits += 1
            return LLMResponse(**cached)

        resp = self._inner.complete(system, user, max_tokens, temperature)
        with self._lock:
            self.misses += 1
            self._cache[key] = resp.model_dump()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._cache), encoding="utf-8")
        return resp


class MeteredBackend(LLMBackend):
    """Accumulate call and token counts across a run (thread-safe)."""

    def __init__(self, inner: LLMBackend):
        self._inner = inner
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def name(self) -> str:
        return self._inner.name

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> LLMResponse:
        resp = self._inner.complete(system, user, max_tokens, temperature)
        with self._lock:
            self.calls += 1
            self.input_tokens += resp.input_tokens or 0
            self.output_tokens += resp.output_tokens or 0
        return resp

    @property
    def usage(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    def cost(self, price_in_per_1k: float, price_out_per_1k: float) -> float:
        """Estimated USD cost given per-1K-token input/output prices."""
        return (
            self.input_tokens / 1000 * price_in_per_1k
            + self.output_tokens / 1000 * price_out_per_1k
        )
