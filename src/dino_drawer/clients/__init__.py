"""Central API clients for dino-drawer.

Exposes:
    GeminiClient — thin wrapper around google-genai with retry and typed errors.
"""
from dino_drawer.clients.gemini import GeminiClient, GeminiError

__all__ = ["GeminiClient", "GeminiError"]
