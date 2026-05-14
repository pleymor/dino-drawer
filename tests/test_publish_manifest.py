"""Tests for dino_drawer.publish.manifest — species entry + catalog helpers."""
from __future__ import annotations

import re

from dino_drawer.models import (
    Dimensions,
    FactSheet,
    Habitat,
    Integument,
    Posture,
    Reference,
    SignatureTraits,
    VisualReferences,
)
from dino_drawer.publish.manifest import (
    build_species_entry,
    remove_from_catalog,
    upsert_catalog,
)


def _make_factsheet(species: str = "Tyrannosaurus rex") -> FactSheet:
    """Return a minimal FactSheet suitable for testing."""
    return FactSheet(
        species=species,
        subtitle="The Tyrant Lizard King",
        dimensions=Dimensions(
            body_length="12-13 m", hip_height="3.5-4 m", skull_length="~1.5 m",
            forelimb_length="~1 m (very short)", tail_length="~6 m",
            body_mass="~8.8 t",
            source_ids=["ref1"],
        ),
        integument=Integument(
            integument_type="scaly", coloration="dark grey dorsal",
            keratinous_structures="claws",
            ontogenetic_variation="no data in corpus",
            source_ids=["ref1"],
        ),
        posture=Posture(
            stance="bipedal", typical_posture="horizontal spine",
            locomotion_mode="cursorial",
            source_ids=["ref1"],
        ),
        habitat=Habitat(
            geological_period="Late Cretaceous, ~68-66 Ma",
            biome="coastal plain", region_or_formation="Hell Creek Formation",
            source_ids=["ref1"],
        ),
        signature_traits=SignatureTraits(
            text="Massive boxy skull, tiny two-fingered forelimbs",
            source_ids=["ref1"],
        ),
        conclusion="An apex predator of the Late Cretaceous.",
        references=[
            Reference(id="ref1", citation_short="Smith 2020", doi=None, title="T. rex anatomy"),
        ],
        visual_references=VisualReferences(),
        image_prompt="A Tyrannosaurus rex in a forest.",
    )


_IMAGE_URL = "https://pub.r2.dev/tyrannosaurus-rex.webp"
_THUMB_URL = "https://pub.r2.dev/tyrannosaurus-rex_thumbnail.webp"


class TestBuildSpeciesEntry:
    """Tests for build_species_entry()."""

    def test_top_level_keys(self) -> None:
        entry = build_species_entry(_make_factsheet(), _IMAGE_URL, _THUMB_URL)
        assert set(entry.keys()) == {
            "slug", "species", "subtitle",
            "dimensions", "integument", "posture", "habitat", "signature_traits",
            "conclusion", "references", "image_prompt",
            "image_url", "thumbnail_url", "generated_at",
        }

    def test_slug_derived_from_species(self) -> None:
        entry = build_species_entry(
            _make_factsheet("Tyrannosaurus rex"), _IMAGE_URL, _THUMB_URL
        )
        assert entry["slug"] == "tyrannosaurus-rex"

    def test_image_url_passed_through(self) -> None:
        entry = build_species_entry(_make_factsheet(), _IMAGE_URL, _THUMB_URL)
        assert entry["image_url"] == _IMAGE_URL

    def test_thumbnail_url_passed_through(self) -> None:
        entry = build_species_entry(_make_factsheet(), _IMAGE_URL, _THUMB_URL)
        assert entry["thumbnail_url"] == _THUMB_URL

    def test_dimensions_serialised(self) -> None:
        entry = build_species_entry(_make_factsheet(), _IMAGE_URL, _THUMB_URL)
        assert isinstance(entry["dimensions"], dict)
        assert entry["dimensions"]["body_mass"].startswith("~")

    def test_signature_traits_serialised(self) -> None:
        entry = build_species_entry(_make_factsheet(), _IMAGE_URL, _THUMB_URL)
        assert "boxy skull" in entry["signature_traits"]["text"]

    def test_generated_at_iso_format(self) -> None:
        entry = build_species_entry(_make_factsheet(), _IMAGE_URL, _THUMB_URL)
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", entry["generated_at"])


class TestUpsertCatalog:
    """Tests for upsert_catalog()."""

    def test_adds_new_entry_to_empty_catalog(self) -> None:
        entry = build_species_entry(_make_factsheet(), _IMAGE_URL, _THUMB_URL)
        catalog = upsert_catalog(None, entry)

        assert catalog["count"] == 1
        assert catalog["species"][0]["slug"] == "tyrannosaurus-rex"

    def test_replaces_existing_entry_with_same_slug(self) -> None:
        entry = build_species_entry(_make_factsheet(), _IMAGE_URL, _THUMB_URL)
        catalog = upsert_catalog(None, entry)
        catalog = upsert_catalog(catalog, entry)
        assert catalog["count"] == 1

    def test_entries_sorted_by_species_name(self) -> None:
        entry_trex = build_species_entry(
            _make_factsheet("Tyrannosaurus rex"), _IMAGE_URL, _THUMB_URL
        )
        entry_vel = build_species_entry(
            _make_factsheet("Velociraptor mongoliensis"),
            "https://pub.r2.dev/velociraptor-mongoliensis.webp",
            "https://pub.r2.dev/velociraptor-mongoliensis_thumbnail.webp",
        )
        catalog = upsert_catalog(None, entry_vel)
        catalog = upsert_catalog(catalog, entry_trex)

        names = [s["species"] for s in catalog["species"]]
        assert names == sorted(names)
        assert names[0] == "Tyrannosaurus rex"


class TestRemoveFromCatalog:
    """Tests for remove_from_catalog()."""

    def test_removes_matching_slug(self) -> None:
        entry = build_species_entry(_make_factsheet(), _IMAGE_URL, _THUMB_URL)
        catalog = upsert_catalog(None, entry)
        catalog = remove_from_catalog(catalog, "tyrannosaurus-rex")
        assert catalog["count"] == 0
        assert catalog["species"] == []

    def test_count_decremented(self) -> None:
        entry_trex = build_species_entry(
            _make_factsheet("Tyrannosaurus rex"), _IMAGE_URL, _THUMB_URL
        )
        entry_vel = build_species_entry(
            _make_factsheet("Velociraptor mongoliensis"),
            "https://pub.r2.dev/velociraptor-mongoliensis.webp",
            "https://pub.r2.dev/velociraptor-mongoliensis_thumbnail.webp",
        )
        catalog = upsert_catalog(upsert_catalog(None, entry_trex), entry_vel)
        assert catalog["count"] == 2
        catalog = remove_from_catalog(catalog, "velociraptor-mongoliensis")
        assert catalog["count"] == 1

    def test_nonexistent_slug_is_noop(self) -> None:
        entry = build_species_entry(_make_factsheet(), _IMAGE_URL, _THUMB_URL)
        catalog = upsert_catalog(None, entry)
        catalog = remove_from_catalog(catalog, "unknown-species")
        assert catalog["count"] == 1
