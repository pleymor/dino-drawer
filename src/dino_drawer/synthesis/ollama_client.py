"""Gemini-backed JSON-mode generator. Module name kept for back-compat with imports."""
from __future__ import annotations

from dino_drawer.clients.gemini import GeminiClient, GeminiError


class SynthesisError(RuntimeError):
    """Raised when the LLM cannot produce valid JSON after retries."""


def call_llm_for_json(
    *,
    model: str | None = None,
    prompt: str,
    max_retries: int = 2,
    temperature: float = 0.2,
) -> dict:
    """Call Gemini in JSON mode and return the parsed dict.

    Args:
        model: Gemini model name. None falls back to GEMINI_TEXT_MODEL env var
            (default ``gemini-2.5-flash``).
        prompt: The fully assembled prompt to send.
        max_retries: Number of retries on empty/non-dict responses.
        temperature: Accepted for back-compat but not currently forwarded
            (Gemini JSON-mode is deterministic enough for our use case).

    Returns:
        The parsed JSON dict.

    Raises:
        SynthesisError: If Gemini fails or returns invalid JSON after retries.
    """
    client = GeminiClient()
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            if attempt == 0:
                data = client.chat_json(prompt, model=model)
            else:
                retry_prompt = (
                    f"{prompt}\n\nPrevious output was invalid ({last_err}). "
                    "Return ONLY the JSON, no markdown."
                )
                data = client.chat_json(retry_prompt, model=model)
        except GeminiError as exc:
            raise SynthesisError(f"Gemini call failed: {exc}") from exc
        if isinstance(data, dict):
            return data
        last_err = "non-dict response"
    raise SynthesisError(f"LLM failed to return valid JSON after {max_retries + 1} attempts")
