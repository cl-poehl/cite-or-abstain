"""Network-free LLM backends for tests and offline development.

The categorizer and verifier only need `LLMBackend.complete()`, so a fake backend
is all it takes to exercise the whole pipeline deterministically — no API key, no
network, no flakiness. These are exported from the package so downstream users can
unit-test their own harness wiring the same way.
"""
from __future__ import annotations

from collections.abc import Callable

from .llm.base import LLMBackend, LLMResponse


class FixedBackend(LLMBackend):
    """Returns the same text for every call. Handy for verifier-alignment tests."""

    def __init__(self, text: str = "supports", model: str = "fake"):
        self._text = text
        self._model = model
        self.calls = 0

    @property
    def name(self) -> str:
        return f"fake/fixed:{self._model}"

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self._text, model=self._model)


class ScriptedBackend(LLMBackend):
    """Returns queued responses in order; repeats the last once exhausted.

    Deterministic and inspectable (`.calls` counts completions). Pass a list of raw
    response strings the fake judge should emit, in call order.
    """

    def __init__(self, responses: list[str], model: str = "fake"):
        self._responses = list(responses)
        self._model = model
        self.calls = 0

    @property
    def name(self) -> str:
        return f"fake/scripted:{self._model}"

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> LLMResponse:
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return LLMResponse(text=text, model=self._model)


class RoutedBackend(LLMBackend):
    """Dispatches to a user function `route(system, user) -> str`.

    Lets a test decide the response from the prompt content — e.g. return a
    categorization JSON when the categorizer prompt is seen and an alignment word
    when the verifier prompt is seen.
    """

    def __init__(self, route: Callable[[str, str], str], model: str = "fake"):
        self._route = route
        self._model = model
        self.calls = 0

    @property
    def name(self) -> str:
        return f"fake/routed:{self._model}"

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self._route(system, user), model=self._model)
