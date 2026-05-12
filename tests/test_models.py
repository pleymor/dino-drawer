"""Tests for Pydantic models used as pipeline artifacts."""
from dino_drawer.models import (
    Paper,
    PapersFile,
    RawImage,
    RawImagesFile,
    ClassifiedImage,
    RefsFile,
    Annotation,
    SkullView,
    Size,
    Reference,
    VisualRef,
    VisualReferences,
    FactSheet,
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


def test_factsheet_requires_each_fact_has_source():
    """Validation: every annotation fact must reference an existing source_id."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FactSheet(
            species="X",
            subtitle="",
            annotations=[Annotation(region="tête", facts=["fact"], source_ids=["missing"])],
            skull_view=SkullView(facts=["f"], scale_cm=10, source_ids=["paper:0"]),
            size=Size(length_m=[1, 2], hip_height_m=[1, 2], source_ids=["paper:0"]),
            conclusion="",
            references=[Reference(id="paper:0", citation_short="X", doi=None, title="t")],
            visual_references=VisualReferences(body=[], skull=[]),
            image_prompt="...",
        )


def test_factsheet_valid():
    fs = FactSheet(
        species="Tyrannosaurus rex",
        subtitle="...",
        annotations=[Annotation(region="tête", facts=["museau"], source_ids=["paper:0"])],
        skull_view=SkullView(facts=["..."], scale_cm=50, source_ids=["paper:0"]),
        size=Size(length_m=[12, 13], hip_height_m=[3.5, 4], source_ids=["paper:0"]),
        conclusion="...",
        references=[Reference(id="paper:0", citation_short="X 2024", doi=None, title="t")],
        visual_references=VisualReferences(body=[], skull=[]),
        image_prompt="photoreal trex, no text",
    )
    assert fs.species == "Tyrannosaurus rex"
