"""Anthropic backend for cite-or-abstain.

Uses the official anthropic Python SDK. Defaults to Claude Sonnet 4.6.
Reads ANTHROPIC_API_KEY from environment unless an explicit api_key is provided.
"""
from __future__ import annotations

import os

from anthropic import Anthropic, BadRequestError

from .base import LLMBackend, LLMResponse

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._model = model
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        # Newer models (e.g. Sonnet 5) reject the `temperature` param. Send it, and if the
        # model refuses, drop it for this and all later calls.
        self._send_temperature = True

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
        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self._send_temperature:
            kwargs["temperature"] = temperature
        try:
            msg = self._client.messages.create(**kwargs)
        except BadRequestError as e:
            if self._send_temperature and "temperature" in str(e).lower():
                self._send_temperature = False
                kwargs.pop("temperature", None)
                msg = self._client.messages.create(**kwargs)
            else:
                raise
        text = "".join(block.text for block in msg.content if hasattr(block, "text"))
        usage = getattr(msg, "usage", None)
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
