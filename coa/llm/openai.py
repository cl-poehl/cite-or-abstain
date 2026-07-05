"""OpenAI backend for cite-or-abstain.

Uses the official openai Python SDK chat completions. Defaults to GPT-4o.
Reads OPENAI_API_KEY from environment unless an explicit api_key is provided.
"""
from __future__ import annotations

import os

from openai import BadRequestError, OpenAI

from .base import LLMBackend, LLMResponse

DEFAULT_MODEL = "gpt-4o"


def _downgrade_params(error_msg: str, kwargs: dict) -> bool:
    """Adjust `kwargs` in place for a model that rejects a param, per the API error text.

    Newer OpenAI models diverge from the chat-completions defaults in two ways: they reject
    a non-default `temperature`, and they replace `max_tokens` with `max_completion_tokens`.
    Returns True if anything changed (so a retry is worthwhile), False otherwise.
    """
    changed = False
    if "temperature" in kwargs and "temperature" in error_msg:
        del kwargs["temperature"]
        changed = True
    if "max_tokens" in kwargs and "max_completion_tokens" in error_msg:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        changed = True
    return changed


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._model = model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        # Learned across calls: newer models (o1/o3/gpt-5) reject `temperature` and rename
        # `max_tokens` -> `max_completion_tokens`. Start with the classic params and
        # downgrade on the first rejection, remembering it for later calls.
        self._send_temperature = True
        self._token_param = "max_tokens"

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
        kwargs: dict = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        kwargs[self._token_param] = max_tokens
        if self._send_temperature:
            kwargs["temperature"] = temperature

        last_exc: BadRequestError | None = None
        resp = None
        for _ in range(3):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except BadRequestError as e:
                last_exc = e
                if not _downgrade_params(str(e).lower(), kwargs):
                    raise
        if resp is None:  # pragma: no cover - only if a model keeps rejecting adjustable params
            raise last_exc  # type: ignore[misc]

        # Remember what worked so later calls skip the rejected params.
        self._send_temperature = "temperature" in kwargs
        self._token_param = (
            "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
        )

        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )
