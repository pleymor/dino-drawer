"""Tests for the diffusion module. Pipeline is mocked; we test wiring only."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from dino_drawer.models import FactSheet, VisualRef, VisualReferences, Annotation, SkullView, Size, Reference
from dino_drawer.image.diffusion import generate_assets


def _basic_factsheet(**overrides) -> FactSheet:
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


def test_generate_assets_uses_ip_adapter_when_body_refs_present(tmp_path):
    img = Image.new("RGB", (64, 64))
    (tmp_path / "refs").mkdir()
    Image.new("RGB", (64, 64)).save(tmp_path / "refs/body_0.jpg")

    fs = _basic_factsheet(visual_references=VisualReferences(
        body=[VisualRef(path="refs/body_0.jpg", credit="x", license="CC0", score=8)],
        skull=[],
    ))

    mock_pipe = MagicMock()
    mock_pipe.return_value = MagicMock(images=[img])

    with patch("dino_drawer.image.diffusion._load_pipeline", return_value=mock_pipe) as loader:
        generate_assets(factsheet=fs, out_dir=tmp_path, model="stabilityai/stable-diffusion-xl-base-1.0")
    call_kwargs = mock_pipe.call_args.kwargs
    assert "ip_adapter_image" in call_kwargs
    assert (tmp_path / "hero.png").exists()
    assert (tmp_path / "silhouette.svg").exists()


def test_generate_assets_runs_text_only_when_no_refs(tmp_path):
    img = Image.new("RGB", (64, 64))
    fs = _basic_factsheet()

    mock_pipe = MagicMock()
    mock_pipe.return_value = MagicMock(images=[img])

    with patch("dino_drawer.image.diffusion._load_pipeline", return_value=mock_pipe):
        generate_assets(factsheet=fs, out_dir=tmp_path, model="stabilityai/stable-diffusion-xl-base-1.0")

    call_kwargs = mock_pipe.call_args.kwargs
    assert "ip_adapter_image" not in call_kwargs or call_kwargs.get("ip_adapter_image") is None
    assert (tmp_path / "hero.png").exists()
