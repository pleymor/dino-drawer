"""Tests for the synthesis Ollama client (JSON validation + retry)."""
from unittest.mock import patch

import pytest

from dino_drawer.synthesis.ollama_client import call_llm_for_json, SynthesisError


def test_returns_parsed_json_on_first_try():
    fake = {"message": {"content": '{"k":"v"}'}}
    with patch("dino_drawer.synthesis.ollama_client.ollama.chat", return_value=fake):
        out = call_llm_for_json(model="qwen2.5:14b", prompt="x")
    assert out == {"k": "v"}


def test_retries_then_succeeds():
    bad = {"message": {"content": "not json"}}
    good = {"message": {"content": '{"k":1}'}}
    with patch("dino_drawer.synthesis.ollama_client.ollama.chat", side_effect=[bad, good]) as m:
        out = call_llm_for_json(model="qwen2.5:14b", prompt="x", max_retries=2)
    assert out == {"k": 1}
    assert m.call_count == 2


def test_raises_after_max_retries():
    bad = {"message": {"content": "nope"}}
    with patch("dino_drawer.synthesis.ollama_client.ollama.chat", return_value=bad):
        with pytest.raises(SynthesisError):
            call_llm_for_json(model="qwen2.5:14b", prompt="x", max_retries=2)
