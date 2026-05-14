"""Tests for Gemini-backed image generation."""
import io
from unittest.mock import patch
from PIL import Image

from dino_drawer.models import (
    FactSheet, VisualRef, VisualReferences, Reference,
    Dimensions, Integument, Posture, Habitat, SignatureTraits,
)
from dino_drawer.image.diffusion import generate_assets


def _png_bytes(color: str = "olive") -> bytes:
    """Return minimal PNG bytes of a 64x64 solid-colour image."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


def _basic_factsheet(**overrides) -> FactSheet:
    """Build a minimal valid FactSheet, optionally overriding any field."""
    base = dict(
        species="Tyrannosaurus rex",
        subtitle="",
        dimensions=Dimensions(
            body_length="x", hip_height="x", skull_length="x",
            forelimb_length="x", tail_length="x", body_mass="x",
            source_ids=["wikipedia"],
        ),
        integument=Integument(
            integument_type="x", coloration="x",
            keratinous_structures="x", ontogenetic_variation="x",
            source_ids=["wikipedia"],
        ),
        posture=Posture(
            stance="x", typical_posture="x", locomotion_mode="x",
            source_ids=["wikipedia"],
        ),
        habitat=Habitat(
            geological_period="x", biome="x", region_or_formation="x",
            source_ids=["wikipedia"],
        ),
        signature_traits=SignatureTraits(text="x", source_ids=["wikipedia"]),
        conclusion="",
        references=[Reference(id="wikipedia", citation_short="W", doi=None, title="t")],
        visual_references=VisualReferences(body=[]),
        image_prompt="photoreal trex no text",
    )
    base.update(overrides)
    return FactSheet(**base)


def test_generate_assets_passes_body_refs_to_gemini(tmp_path):
    """Hero call receives the body ref path."""
    (tmp_path / "refs").mkdir()
    body_path = tmp_path / "refs/body_0.jpg"
    Image.new("RGB", (64, 64)).save(body_path, "JPEG")

    fs = _basic_factsheet(visual_references=VisualReferences(
        body=[VisualRef(path="refs/body_0.jpg", credit="x", license="CC0", score=8)],
    ))

    with patch("dino_drawer.image.diffusion.GeminiClient") as MockClient:
        MockClient.return_value.generate_image.return_value = _png_bytes()
        generate_assets(factsheet=fs, out_dir=tmp_path)

    calls = MockClient.return_value.generate_image.call_args_list
    assert len(calls) == 1
    assert calls[0].kwargs.get("refs") == [body_path]

    assert (tmp_path / "hero.png").exists()


def test_generate_assets_text_only_when_no_refs(tmp_path):
    """Single hero call with empty refs when factsheet has no visual refs."""
    fs = _basic_factsheet()
    with patch("dino_drawer.image.diffusion.GeminiClient") as MockClient:
        MockClient.return_value.generate_image.return_value = _png_bytes()
        generate_assets(factsheet=fs, out_dir=tmp_path)

    calls = MockClient.return_value.generate_image.call_args_list
    assert len(calls) == 1
    assert calls[0].kwargs.get("refs") == []

    assert (tmp_path / "hero.png").exists()
