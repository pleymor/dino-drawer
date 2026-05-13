"""Tests for the Gemini-backed VLM client."""
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from dino_drawer.vision.vlm_client import VLMClient, VLMError


def _write_jpeg(path: Path, w: int = 64, h: int = 64) -> None:
    """Write a small valid JPEG for use as test input."""
    Image.new("RGB", (w, h), "olive").save(path, "JPEG")


def test_classify_returns_parsed_json(tmp_path):
    img = tmp_path / "img.jpg"
    _write_jpeg(img)
    good = {"type": "paleoart_realiste", "view": "profil_corps",
            "usable_for_body_generation": True, "usable_for_skull_generation": False,
            "realism_score": 8, "quality_score": 7, "description_courte": "x"}
    with patch("dino_drawer.vision.vlm_client.GeminiClient") as MockClient:
        MockClient.return_value.chat_json.return_value = good
        out = VLMClient().classify_image(img, species="Tyrannosaurus rex")
    assert out["type"] == "paleoart_realiste"


def test_classify_retries_on_invalid_then_succeeds(tmp_path):
    img = tmp_path / "img.jpg"
    _write_jpeg(img)
    bad = {"foo": "bar"}  # no 'type' key
    good = {"type": "autre", "view": "autre",
            "usable_for_body_generation": False, "usable_for_skull_generation": False,
            "realism_score": 0, "quality_score": 0, "description_courte": ""}
    with patch("dino_drawer.vision.vlm_client.GeminiClient") as MockClient:
        MockClient.return_value.chat_json.side_effect = [bad, good]
        out = VLMClient(max_retries=1).classify_image(img, species="X")
    assert out["type"] == "autre"


def test_classify_raises_after_max_retries(tmp_path):
    img = tmp_path / "img.jpg"
    _write_jpeg(img)
    bad = {"foo": "bar"}
    with patch("dino_drawer.vision.vlm_client.GeminiClient") as MockClient:
        MockClient.return_value.chat_json.return_value = bad
        with pytest.raises(VLMError):
            VLMClient(max_retries=1).classify_image(img, species="X")


def test_describe_image_returns_stripped_text(tmp_path):
    img = tmp_path / "img.jpg"
    _write_jpeg(img)
    with patch("dino_drawer.vision.vlm_client.GeminiClient") as MockClient:
        MockClient.return_value.chat.return_value = "  some description  "
        out = VLMClient().describe_image(img, prompt="describe this")
    assert out == "some description"
