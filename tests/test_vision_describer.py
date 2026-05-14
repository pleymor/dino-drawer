"""Tests for VLM-based body-reference descriptions."""
from pathlib import Path
from unittest.mock import patch

from dino_drawer.models import ClassifiedImage, RefsFile
from dino_drawer.vision.describer import describe_body_refs


def _classified(i: int) -> ClassifiedImage:
    """Create a test ClassifiedImage with index i."""
    return ClassifiedImage(
        id=i,
        path=f"refs/body_{i}.jpg",
        type="paleoart_realiste",
        view="profil_corps",
        usable_for_body_generation=True,
        realism_score=8,
        quality_score=8,
        description_courte="",
        credit="x",
        license="CC0",
    )


def test_describer_returns_concatenated_brief(tmp_path):
    """Test that describe_body_refs calls VLM for each body ref and concatenates results."""
    (tmp_path / "refs").mkdir()
    for i in range(3):
        (tmp_path / f"refs/body_{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    refs = RefsFile(
        species="X",
        body=[_classified(0), _classified(1), _classified(2)],
    )

    with patch("dino_drawer.vision.describer.VLMClient") as MockVLM:
        MockVLM.return_value.describe_image.side_effect = [
            "Green skin with dark stripes. Sparse feathers on the neck.",
            "S-shaped posture, mouth slightly ajar.",
            "Humid forest, slanted light.",
        ]
        brief = describe_body_refs(refs=refs, out_dir=tmp_path)

    assert "green skin" in brief.lower()
    assert "feathers" in brief.lower()
    assert "forest" in brief.lower()
