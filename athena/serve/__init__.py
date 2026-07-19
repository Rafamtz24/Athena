"""Serve mode — expose the resident reasoning model over an OpenAI-compatible API.

Lets external clients (Open WebUI, LM Studio-compatible tools, etc.) use
Athena's already-loaded reasoning model without loading a second copy. The
Thought pipeline, tools, memory, and learning are all bypassed — serve mode is
pure model inference.
"""

from athena.serve.openai_server import build_app, can_serve

__all__ = ["build_app", "can_serve"]
