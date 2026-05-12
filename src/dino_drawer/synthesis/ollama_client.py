"""Wrapper around Ollama for JSON-mode generation with retry-on-invalid."""
from __future__ import annotations

import json
import re

import ollama


class SynthesisError(RuntimeError):
    """Raised when the LLM cannot produce valid JSON after retries."""


def _extract_json(text: str) -> dict | None:
    """
    Attempt to extract and parse JSON from text.

    First tries parsing the entire text as JSON. If that fails,
    searches for a JSON object pattern and tries to parse that.

    Args:
        text: The text to extract JSON from.

    Returns:
        Parsed dict if valid JSON found, None otherwise.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_llm_for_json(
    *,
    model: str,
    prompt: str,
    max_retries: int = 2,
    temperature: float = 0.2,
) -> dict:
    """
    Call the LLM expecting JSON, retry up to max_retries on parse failure.

    Args:
        model: The Ollama model identifier (e.g., "qwen2.5:14b").
        prompt: The initial prompt to send to the LLM.
        max_retries: Number of retries after initial attempt. Total attempts = max_retries + 1.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        Parsed JSON response as a dict.

    Raises:
        SynthesisError: If valid JSON cannot be obtained after all retry attempts.
    """
    last_err = ""
    for attempt in range(max_retries + 1):
        user = prompt if attempt == 0 else (
            f"{prompt}\n\nLa précédente sortie était invalide ({last_err}). "
            "Renvoie uniquement le JSON, sans markdown, sans texte avant ou après."
        )
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": user}],
            options={"temperature": temperature},
            format="json",
        )
        content = resp.get("message", {}).get("content", "")
        parsed = _extract_json(content)
        if parsed is not None:
            return parsed
        last_err = "unparseable JSON"
    raise SynthesisError(f"LLM failed to return valid JSON after {max_retries + 1} attempts")
