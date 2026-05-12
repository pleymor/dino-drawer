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
        usable_for_skull_generation=False,
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
        skull=[],
    )

    with patch("dino_drawer.vision.describer.VLMClient") as MockVLM:
        MockVLM.return_value.describe_image.side_effect = [
            "Peau verte avec rayures sombres. Plumes éparses sur le cou.",
            "Posture en S, gueule entrouverte.",
            "Forêt humide, lumière oblique.",
        ]
        brief = describe_body_refs(refs=refs, out_dir=tmp_path)

    assert "peau verte" in brief.lower() or "Peau verte" in brief
    assert "plumes" in brief.lower()
    assert "forêt" in brief.lower() or "forêt" in brief
