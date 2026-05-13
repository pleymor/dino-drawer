"""Tests for the Gemini-backed synthesis JSON client."""
from unittest.mock import patch

import pytest

from dino_drawer.synthesis.ollama_client import call_llm_for_json, SynthesisError


def test_returns_parsed_json_on_first_try():
    with patch("dino_drawer.synthesis.ollama_client.GeminiClient") as MockClient:
        MockClient.return_value.chat_json.return_value = {"k": "v"}
        out = call_llm_for_json(model="gemini-2.5-flash", prompt="x")
    assert out == {"k": "v"}


def test_retries_then_succeeds():
    with patch("dino_drawer.synthesis.ollama_client.GeminiClient") as MockClient:
        MockClient.return_value.chat_json.side_effect = ["not a dict", {"k": 1}]
        out = call_llm_for_json(model="gemini-2.5-flash", prompt="x", max_retries=2)
    assert out == {"k": 1}


def test_raises_after_max_retries():
    with patch("dino_drawer.synthesis.ollama_client.GeminiClient") as MockClient:
        MockClient.return_value.chat_json.return_value = "not a dict"
        with pytest.raises(SynthesisError):
            call_llm_for_json(model="gemini-2.5-flash", prompt="x", max_retries=2)
