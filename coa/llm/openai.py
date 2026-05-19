"""OpenAI backend for cite-or-abstain.

Uses the official openai Python SDK chat completions. Defaults to GPT-4o.
Reads OPENAI_API_KEY from environment unless an explicit api_key is provided.
"""
from __future__ import annotations

import os

from openai import OpenAI

from .base import LLMBackend, LLMResponse

DEFAULT_MODEL = "gpt-4o"


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._model = model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, model=self._model)
