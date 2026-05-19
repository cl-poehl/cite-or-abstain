"""Anthropic backend for cite-or-abstain.

Uses the official anthropic Python SDK. Defaults to Claude Sonnet 4.6.
Reads ANTHROPIC_API_KEY from environment unless an explicit api_key is provided.
"""
from __future__ import annotations

import os

from anthropic import Anthropic

from .base import LLMBackend, LLMResponse

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._model = model
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in msg.content if hasattr(block, "text"))
        return LLMResponse(text=text, model=self._model)
