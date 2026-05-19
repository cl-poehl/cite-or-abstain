"""Abstract base class for LLM backends.

Backends return raw text; parsing into structured types happens in
coa.categorizer and coa.verifier. This keeps backends thin and the
prompt/parse logic in one place.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    model: str


class LLMBackend(ABC):
    """Minimal LLM-completion interface used by the categorizer and verifier."""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Return a completion. Implementations should default to deterministic (temp 0)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier, e.g., 'anthropic/claude-sonnet-4-6'."""
