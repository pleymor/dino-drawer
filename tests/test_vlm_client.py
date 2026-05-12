"""Tests for the VLM (vision LLM) Ollama wrapper."""
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from dino_drawer.vision.vlm_client import VLMClient, VLMError


def _write_jpeg(path: Path, w: int = 64, h: int = 64) -> None:
    """Write a small valid JPEG so the downscale step can decode it."""
    Image.new("RGB", (w, h), "olive").save(path, "JPEG")


def test_classify_returns_parsed_json(tmp_path):
    img = tmp_path / "img.jpg"
    _write_jpeg(img)

    fake_response = {
        "message": {"content": '{"type":"paleoart_realiste","view":"profil_corps","usable_for_body_generation":true,"usable_for_skull_generation":false,"realism_score":8,"quality_score":7,"description_courte":"trex de profil"}'}
    }

    with patch("dino_drawer.vision.vlm_client.ollama.chat", return_value=fake_response) as m:
        client = VLMClient(model="qwen2.5vl:7b")
        out = client.classify_image(img, species="Tyrannosaurus rex")
    assert out["type"] == "paleoart_realiste"
    assert out["usable_for_body_generation"] is True
    m.assert_called_once()


def test_classify_retries_on_invalid_json(tmp_path):
    img = tmp_path / "img.jpg"
    _write_jpeg(img)
    bad = {"message": {"content": "not json at all"}}
    good = {"message": {"content": '{"type":"autre","view":"autre","usable_for_body_generation":false,"usable_for_skull_generation":false,"realism_score":0,"quality_score":0,"description_courte":""}'}}
    with patch("dino_drawer.vision.vlm_client.ollama.chat", side_effect=[bad, good]) as m:
        client = VLMClient(model="qwen2.5vl:7b")
        out = client.classify_image(img, species="X")
    assert m.call_count == 2
    assert out["type"] == "autre"


def test_classify_raises_after_max_retries(tmp_path):
    img = tmp_path / "img.jpg"
    _write_jpeg(img)
    with patch("dino_drawer.vision.vlm_client.ollama.chat", return_value={"message": {"content": "garbage"}}):
        client = VLMClient(model="qwen2.5vl:7b", max_retries=2)
        with pytest.raises(VLMError):
            client.classify_image(img, species="X")
