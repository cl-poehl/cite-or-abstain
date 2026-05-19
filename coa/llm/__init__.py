"""Pluggable LLM backends for cite-or-abstain.

Two backends bundled: Anthropic and OpenAI. Adding a new backend is a
~20-line subclass of LLMBackend.
"""
from .anthropic import AnthropicBackend
from .base import LLMBackend, LLMResponse
from .openai import OpenAIBackend

__all__ = ["AnthropicBackend", "LLMBackend", "LLMResponse", "OpenAIBackend"]
