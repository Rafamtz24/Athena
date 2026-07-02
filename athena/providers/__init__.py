"""
Athena Providers Package

Contains provider implementations for LLM inference backends.
Supports multiple providers: LM Studio, local GGUF (llama.cpp),
OpenRouter, Ollama, and others.

Use ProviderFactory to create provider instances.
The Brain must never instantiate providers directly.
"""

from athena.providers.factory import ProviderFactory

__all__ = ["ProviderFactory"]