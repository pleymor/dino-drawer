"""Tests for the image classifier selection rules (VLM mocked)."""
from pathlib import Path
from unittest.mock import patch

from dino_drawer.models import RawImage, RawImagesFile
from dino_drawer.vision.classifier import classify_and_select


def _raw(id_: int, path: str = "refs_raw/x.jpg") -> RawImage:
    return RawImage(
        id=id_, path=path, source_url="http://x",
        source="wikimedia_commons", credit="X", license="CC0",
        width=1200, height=800, search_query="x",
    )


def test_keeps_top_three_body(tmp_path):
    """Top 3 body candidates ranked by realism * quality are kept; skull-only refs dropped."""
    raws = RawImagesFile(species="X", images=[_raw(i, f"refs_raw/i{i}.jpg") for i in range(5)])
    (tmp_path / "refs_raw.json").write_text(raws.model_dump_json())
    raw_dir = tmp_path / "refs_raw"
    raw_dir.mkdir()
    for i in range(5):
        (raw_dir / f"i{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    classifications = [
        {"type": "paleoart_realiste", "view": "profil_corps", "usable_for_body_generation": True, "realism_score": 9, "quality_score": 9, "description_courte": "a"},
        {"type": "paleoart_realiste", "view": "profil_corps", "usable_for_body_generation": True, "realism_score": 7, "quality_score": 8, "description_courte": "b"},
        {"type": "paleoart_realiste", "view": "trois_quarts_corps", "usable_for_body_generation": True, "realism_score": 8, "quality_score": 7, "description_courte": "c"},
        {"type": "paleoart_realiste", "view": "profil_corps", "usable_for_body_generation": True, "realism_score": 5, "quality_score": 5, "description_courte": "d"},
        # Not a body view — should be ignored.
        {"type": "photo_squelette", "view": "detail", "usable_for_body_generation": False, "realism_score": 0, "quality_score": 8, "description_courte": "e"},
    ]

    # VLM is invoked from a thread pool, so map the mock by image path
    # (path stem is "i{idx}.jpg") rather than relying on call order.
    def _classify_by_path(image_path, species):  # noqa: ARG001
        idx = int(Path(image_path).stem.removeprefix("i"))
        return classifications[idx]

    with patch("dino_drawer.vision.classifier.VLMClient") as MockVLM:
        instance = MockVLM.return_value
        instance.classify_image.side_effect = _classify_by_path
        refs = classify_and_select(species="X", out_dir=tmp_path)

    assert len(refs.body) == 3
    assert {r.id for r in refs.body} == {0, 1, 2}


def test_drops_rejected_types(tmp_path):
    raws = RawImagesFile(species="X", images=[_raw(0), _raw(1)])
    (tmp_path / "refs_raw.json").write_text(raws.model_dump_json())
    (tmp_path / "refs_raw").mkdir()
    (tmp_path / "refs_raw" / "x.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    classifications = [
        {"type": "cladogramme", "view": "autre", "usable_for_body_generation": False, "realism_score": 0, "quality_score": 0, "description_courte": "x"},
        {"type": "illustration_enfant", "view": "profil_corps", "usable_for_body_generation": True, "realism_score": 9, "quality_score": 9, "description_courte": "x"},
    ]
    with patch("dino_drawer.vision.classifier.VLMClient") as MockVLM:
        MockVLM.return_value.classify_image.side_effect = classifications
        refs = classify_and_select(species="X", out_dir=tmp_path)
    assert refs.body == []
    assert refs.rejected_count == 2
