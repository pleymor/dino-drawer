"""Tests for the central Gemini API client."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from dino_drawer.clients.gemini import GeminiClient, GeminiError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_response(text: str) -> MagicMock:
    """Build a mock response with a .text attribute."""
    resp = MagicMock()
    resp.text = text
    return resp


def _make_image_response(png_bytes: bytes) -> MagicMock:
    """Build a mock response whose first candidate part contains inline_data."""
    part = MagicMock()
    part.inline_data = MagicMock()
    part.inline_data.data = png_bytes
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    resp = MagicMock()
    resp.candidates = [candidate]
    return resp


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------

class TestChat:
    def test_returns_text_response(self):
        """chat() returns the .text from the model response."""
        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.return_value = _make_text_response("hello dino")

            client = GeminiClient()
            result = client.chat("describe a T-Rex")

        assert result == "hello dino"
        mock_models.generate_content.assert_called_once()

    def test_uses_default_text_model(self):
        """chat() uses GEMINI_TEXT_MODEL default when model=None."""
        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.return_value = _make_text_response("ok")

            client = GeminiClient()
            client.chat("hi")

        call_kwargs = mock_models.generate_content.call_args
        assert "gemini-2.5-flash" in str(call_kwargs)

    def test_passes_custom_model(self):
        """chat() forwards an explicitly supplied model name."""
        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.return_value = _make_text_response("ok")

            client = GeminiClient()
            client.chat("hi", model="gemini-custom")

        call_kwargs = mock_models.generate_content.call_args
        assert "gemini-custom" in str(call_kwargs)

    def test_retries_on_429(self):
        """chat() retries up to 4 times on 429 ResourceExhausted errors."""
        from google.api_core.exceptions import ResourceExhausted

        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient, \
             patch("dino_drawer.clients.gemini.time.sleep") as mock_sleep:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.side_effect = [
                ResourceExhausted("quota"),
                ResourceExhausted("quota"),
                _make_text_response("recovered"),
            ]

            client = GeminiClient()
            result = client.chat("hi")

        assert result == "recovered"
        assert mock_sleep.call_count == 2
        # Verify exponential backoff: 1s then 2s
        assert mock_sleep.call_args_list[0] == call(1)
        assert mock_sleep.call_args_list[1] == call(2)

    def test_retries_on_5xx(self):
        """chat() retries on transient 5xx (ServiceUnavailable) errors."""
        from google.api_core.exceptions import ServiceUnavailable

        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient, \
             patch("dino_drawer.clients.gemini.time.sleep"):
            mock_models = MockClient.return_value.models
            mock_models.generate_content.side_effect = [
                ServiceUnavailable("service down"),
                _make_text_response("ok"),
            ]

            client = GeminiClient()
            result = client.chat("hi")

        assert result == "ok"

    def test_raises_gemini_error_after_max_retries(self):
        """chat() raises GeminiError after exhausting all retries."""
        from google.api_core.exceptions import ResourceExhausted

        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient, \
             patch("dino_drawer.clients.gemini.time.sleep"):
            mock_models = MockClient.return_value.models
            mock_models.generate_content.side_effect = ResourceExhausted("quota")

            client = GeminiClient()
            with pytest.raises(GeminiError, match="rate.limit|quota|ResourceExhausted"):
                client.chat("hi")

    def test_raises_gemini_error_on_auth_failure(self):
        """chat() raises GeminiError on permanent auth errors without retrying."""
        from google.api_core.exceptions import PermissionDenied

        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient, \
             patch("dino_drawer.clients.gemini.time.sleep") as mock_sleep:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.side_effect = PermissionDenied("bad key")

            client = GeminiClient()
            with pytest.raises(GeminiError, match="auth|permission|API key"):
                client.chat("hi")

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# chat_json()
# ---------------------------------------------------------------------------

class TestChatJson:
    def test_parses_json_response(self):
        """chat_json() parses the JSON string from the model response."""
        payload = {"species": "T-Rex", "era": "Cretaceous"}
        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.return_value = _make_text_response(
                json.dumps(payload)
            )

            client = GeminiClient()
            result = client.chat_json("describe in JSON")

        assert result == payload

    def test_uses_json_response_mime_type(self):
        """chat_json() passes response_mime_type=application/json in config."""
        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.return_value = _make_text_response('{"ok": true}')

            client = GeminiClient()
            client.chat_json("go")

        call_kwargs = mock_models.generate_content.call_args
        assert "application/json" in str(call_kwargs)

    def test_retries_and_raises_on_persistent_invalid_json(self):
        """chat_json() raises GeminiError after max retries on bad JSON."""
        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient, \
             patch("dino_drawer.clients.gemini.time.sleep"):
            mock_models = MockClient.return_value.models
            mock_models.generate_content.return_value = _make_text_response(
                "not json at all"
            )

            client = GeminiClient()
            with pytest.raises(GeminiError, match="[Jj][Ss][Oo][Nn]|parse"):
                client.chat_json("give me json")


# ---------------------------------------------------------------------------
# generate_image()
# ---------------------------------------------------------------------------

class TestGenerateImage:
    def test_returns_png_bytes(self):
        """generate_image() extracts and returns inline_data bytes."""
        fake_png = b"\x89PNG\r\n\x1a\n fake png data"
        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.return_value = _make_image_response(fake_png)

            client = GeminiClient()
            result = client.generate_image("draw a velociraptor")

        assert result == fake_png

    def test_uses_default_image_model(self):
        """generate_image() uses GEMINI_IMAGE_MODEL default when model=None."""
        fake_png = b"PNG"
        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.return_value = _make_image_response(fake_png)

            client = GeminiClient()
            client.generate_image("draw")

        call_kwargs = mock_models.generate_content.call_args
        assert "gemini-3-pro-image-preview" in str(call_kwargs)

    def test_raises_gemini_error_when_no_image_in_response(self):
        """generate_image() raises GeminiError if no inline_data part is found."""
        part = MagicMock()
        part.inline_data = None
        content = MagicMock()
        content.parts = [part]
        candidate = MagicMock()
        candidate.content = content
        resp = MagicMock()
        resp.candidates = [candidate]

        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.return_value = resp

            client = GeminiClient()
            with pytest.raises(GeminiError, match="[Ii]mage|inline_data"):
                client.generate_image("draw")

    def test_retries_on_429_and_returns_image(self):
        """generate_image() retries on ResourceExhausted and succeeds."""
        from google.api_core.exceptions import ResourceExhausted

        fake_png = b"PNG"
        with patch("dino_drawer.clients.gemini.genai.Client") as MockClient, \
             patch("dino_drawer.clients.gemini.time.sleep") as mock_sleep:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.side_effect = [
                ResourceExhausted("quota"),
                _make_image_response(fake_png),
            ]

            client = GeminiClient()
            result = client.generate_image("draw")

        assert result == fake_png
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args_list[0] == call(1)
