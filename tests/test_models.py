"""Tests for Pydantic models used as pipeline artifacts."""
import pytest
from pydantic import ValidationError

from dino_drawer.models import (
    Paper,
    Dimensions,
    Integument,
    Posture,
    Habitat,
    SignatureTraits,
    Reference,
    VisualReferences,
    FactSheet,
)


def _basic_blocks() -> dict:
    """Return a minimal set of mandatory blocks all citing ``paper:0``."""
    return dict(
        dimensions=Dimensions(
            body_length="12-13 m [paper:0]",
            hip_height="3.5-4 m",
            skull_length="~1.5 m",
            forelimb_length="~1 m (very short)",
            tail_length="~6 m",
            body_mass="~8.8 t",
            source_ids=["paper:0"],
        ),
        integument=Integument(
            integument_type="predominantly scaly",
            coloration="dark grey dorsal, lighter ventral",
            keratinous_structures="claws on hands and feet",
            ontogenetic_variation="no data in corpus",
            source_ids=["paper:0"],
        ),
        posture=Posture(
            stance="bipedal",
            typical_posture="horizontal spine, tail as counterweight",
            locomotion_mode="cursorial",
            source_ids=["paper:0"],
        ),
        habitat=Habitat(
            geological_period="Late Cretaceous, Maastrichtian (~68-66 Ma)",
            biome="conifer forest, coastal plain",
            region_or_formation="Hell Creek Formation, Laramidia",
            source_ids=["paper:0"],
        ),
        signature_traits=SignatureTraits(
            text="Massive boxy skull, tiny two-fingered forelimbs",
            source_ids=["paper:0"],
        ),
    )


def test_paper_roundtrip():
    p = Paper(
        doi="10.1234/x",
        title="A study",
        authors=["A. Author"],
        year=2024,
        abstract="...",
        source="semantic_scholar",
    )
    assert Paper.model_validate(p.model_dump()) == p


def test_factsheet_rejects_unknown_source_id():
    """Validation: every block source_id must reference a known Reference id."""
    blocks = _basic_blocks()
    blocks["dimensions"] = Dimensions(
        body_length="x", hip_height="x", skull_length="x", forelimb_length="x",
        tail_length="x", body_mass="x",
        source_ids=["missing_source"],
    )
    with pytest.raises(ValidationError):
        FactSheet(
            species="X",
            subtitle="",
            conclusion="",
            references=[Reference(id="paper:0", citation_short="X", doi=None, title="t")],
            visual_references=VisualReferences(body=[]),
            image_prompt="...",
            **blocks,
        )


def test_factsheet_valid():
    fs = FactSheet(
        species="Tyrannosaurus rex",
        subtitle="...",
        conclusion="...",
        references=[Reference(id="paper:0", citation_short="X 2024", doi=None, title="t")],
        visual_references=VisualReferences(body=[]),
        image_prompt="photoreal trex, no text",
        **_basic_blocks(),
    )
    assert fs.species == "Tyrannosaurus rex"
    assert fs.dimensions.body_mass.startswith("~")
    assert fs.signature_traits.text


def test_factsheet_empty_source_ids_allowed_when_no_data():
    """A block can have empty source_ids if its values are 'no data in corpus'."""
    blocks = _basic_blocks()
    blocks["habitat"] = Habitat(
        geological_period="no data in corpus",
        biome="no data in corpus",
        region_or_formation="no data in corpus",
        source_ids=[],
    )
    fs = FactSheet(
        species="X",
        subtitle="",
        conclusion="",
        references=[Reference(id="paper:0", citation_short="X", doi=None, title="t")],
        visual_references=VisualReferences(body=[]),
        image_prompt="...",
        **blocks,
    )
    assert fs.habitat.source_ids == []
