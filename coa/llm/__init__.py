"""Pluggable LLM backends for cite-or-abstain.

Two backends bundled: Anthropic and OpenAI, each a thin wrapper over the vendor SDK.
The SDKs are **optional extras** (`pip install cite-or-abstain[anthropic]` / `[openai]`
/ `[all]`) and are imported **lazily on first use**, so importing `coa` never pulls a
vendor SDK you are not using — an on-prem-only deployment can install just the one it
needs. Adding a new backend is a ~20-line subclass of `LLMBackend`.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .base import LLMBackend, LLMResponse

if TYPE_CHECKING:  # for type checkers / IDEs only — no runtime SDK import
    from .anthropic import AnthropicBackend
    from .openai import OpenAIBackend

__all__ = ["AnthropicBackend", "LLMBackend", "LLMResponse", "OpenAIBackend"]

_LAZY = {"AnthropicBackend": ".anthropic", "OpenAIBackend": ".openai"}


def __getattr__(name: str) -> object:
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        mod = import_module(submodule, __name__)
    except ModuleNotFoundError as exc:  # the optional vendor SDK is not installed
        extra = "anthropic" if name == "AnthropicBackend" else "openai"
        raise ModuleNotFoundError(
            f"{name} needs the optional '{extra}' extra: "
            f"pip install 'cite-or-abstain[{extra}]'"
        ) from exc
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)
