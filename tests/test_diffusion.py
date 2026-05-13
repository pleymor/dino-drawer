"""Tests for Gemini-backed image generation."""
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import io

from dino_drawer.models import (
    FactSheet, VisualRef, VisualReferences, Annotation, SkullView, Size, Reference,
)
from dino_drawer.image.diffusion import generate_assets


def _png_bytes(color: str = "olive") -> bytes:
    """Return minimal PNG bytes of a 64x64 solid-colour image.

    Args:
        color: PIL colour name for the solid fill.

    Returns:
        Raw PNG bytes.
    """
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


def _basic_factsheet(**overrides) -> FactSheet:
    """Build a minimal valid FactSheet, optionally overriding any field.

    Args:
        **overrides: Fields to override in the base dict.

    Returns:
        A constructed :class:`~dino_drawer.models.FactSheet`.
    """
    base = dict(
        species="Tyrannosaurus rex",
        subtitle="",
        annotations=[Annotation(region="tête", facts=["x"], source_ids=["wikipedia"])],
        skull_view=SkullView(facts=["x"], scale_cm=50, source_ids=["wikipedia"]),
        size=Size(length_m=[12, 13], hip_height_m=[3.5, 4], source_ids=["wikipedia"]),
        conclusion="",
        references=[Reference(id="wikipedia", citation_short="W", doi=None, title="t")],
        visual_references=VisualReferences(body=[], skull=[]),
        image_prompt="photoreal trex no text",
    )
    base.update(overrides)
    return FactSheet(**base)


def test_generate_assets_passes_body_refs_to_gemini(tmp_path):
    """Hero call receives the body ref path; skull call gets empty refs."""
    (tmp_path / "refs").mkdir()
    body_path = tmp_path / "refs/body_0.jpg"
    Image.new("RGB", (64, 64)).save(body_path, "JPEG")

    fs = _basic_factsheet(visual_references=VisualReferences(
        body=[VisualRef(path="refs/body_0.jpg", credit="x", license="CC0", score=8)],
        skull=[],
    ))

    with patch("dino_drawer.image.diffusion.GeminiClient") as MockClient:
        MockClient.return_value.generate_image.return_value = _png_bytes()
        generate_assets(factsheet=fs, out_dir=tmp_path)

    # First call is hero — should have the body ref
    calls = MockClient.return_value.generate_image.call_args_list
    assert len(calls) == 2
    hero_kwargs = calls[0].kwargs
    assert "refs" in hero_kwargs
    assert hero_kwargs["refs"] == [body_path]
    # Skull call: empty refs
    skull_kwargs = calls[1].kwargs
    assert skull_kwargs["refs"] == []

    assert (tmp_path / "hero.png").exists()
    assert (tmp_path / "skull.png").exists()
    assert (tmp_path / "silhouette.svg").exists()


def test_generate_assets_text_only_when_no_refs(tmp_path):
    """Both generate_image calls use empty refs when factsheet has no visual refs."""
    fs = _basic_factsheet()
    with patch("dino_drawer.image.diffusion.GeminiClient") as MockClient:
        MockClient.return_value.generate_image.return_value = _png_bytes()
        generate_assets(factsheet=fs, out_dir=tmp_path)

    for call in MockClient.return_value.generate_image.call_args_list:
        assert call.kwargs.get("refs") == []

    assert (tmp_path / "hero.png").exists()
    assert (tmp_path / "skull.png").exists()
    assert (tmp_path / "silhouette.svg").exists()
