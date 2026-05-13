"""Tests for dino_drawer.publish.manifest — meta.json and catalog.json helpers."""
from __future__ import annotations

from dino_drawer.models import (
    Annotation,
    FactSheet,
    Reference,
    Size,
    SkullView,
    VisualReferences,
)


def _make_factsheet(species: str = "Tyrannosaurus rex") -> FactSheet:
    """Return a minimal FactSheet suitable for testing."""
    return FactSheet(
        species=species,
        subtitle="The Tyrant Lizard King",
        annotations=[
            Annotation(
                region="tête",
                facts=["Large skull"],
                source_ids=["ref1"],
            )
        ],
        skull_view=SkullView(facts=["Robust teeth"], scale_cm=120.0, source_ids=["ref1"]),
        size=Size(length_m=[11.0, 13.0], hip_height_m=[3.0, 4.0], source_ids=["ref1"]),
        conclusion="An apex predator of the Late Cretaceous.",
        references=[
            Reference(id="ref1", citation_short="Smith 2020", doi=None, title="T. rex anatomy")
        ],
        visual_references=VisualReferences(),
        image_prompt="A Tyrannosaurus rex in a forest.",
    )


_IMAGE_URLS: dict[str, dict[int, str]] = {
    "hero": {
        1600: "https://pub.r2.dev/tyrannosaurus-rex/hero@1600.webp",
        800: "https://pub.r2.dev/tyrannosaurus-rex/hero@800.webp",
        400: "https://pub.r2.dev/tyrannosaurus-rex/hero@400.webp",
    },
    "skull": {
        1600: "https://pub.r2.dev/tyrannosaurus-rex/skull@1600.webp",
        800: "https://pub.r2.dev/tyrannosaurus-rex/skull@800.webp",
        400: "https://pub.r2.dev/tyrannosaurus-rex/skull@400.webp",
    },
}


class TestBuildMeta:
    """Tests for build_meta()."""

    def test_correct_top_level_keys(self) -> None:
        """build_meta result should contain all expected top-level keys."""
        from dino_drawer.publish.manifest import build_meta

        fs = _make_factsheet()
        meta = build_meta(fs, _IMAGE_URLS)

        expected_keys = {
            "slug", "species", "subtitle", "annotations", "skull_view",
            "size", "conclusion", "references", "image_prompt", "images", "generated_at",
        }
        assert expected_keys == set(meta.keys())

    def test_slug_derived_from_species(self) -> None:
        """Slug should be lower-case with spaces replaced by hyphens."""
        from dino_drawer.publish.manifest import build_meta

        meta = build_meta(_make_factsheet("Tyrannosaurus rex"), _IMAGE_URLS)
        assert meta["slug"] == "tyrannosaurus-rex"

    def test_images_nested_by_kind_and_width(self) -> None:
        """images dict should map kind → str(width) → url."""
        from dino_drawer.publish.manifest import build_meta

        meta = build_meta(_make_factsheet(), _IMAGE_URLS)
        assert meta["images"]["hero"]["1600"] == _IMAGE_URLS["hero"][1600]
        assert meta["images"]["skull"]["400"] == _IMAGE_URLS["skull"][400]

    def test_annotations_serialised(self) -> None:
        """Annotations should be serialised as plain dicts."""
        from dino_drawer.publish.manifest import build_meta

        meta = build_meta(_make_factsheet(), _IMAGE_URLS)
        assert isinstance(meta["annotations"], list)
        assert meta["annotations"][0]["region"] == "tête"

    def test_generated_at_iso_format(self) -> None:
        """generated_at should be an ISO-8601 UTC timestamp string."""
        from dino_drawer.publish.manifest import build_meta
        import re

        meta = build_meta(_make_factsheet(), _IMAGE_URLS)
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", meta["generated_at"])


class TestUpsertCatalog:
    """Tests for upsert_catalog()."""

    def test_adds_new_entry_to_empty_catalog(self) -> None:
        """upsert_catalog with no existing catalog should create one with one entry."""
        from dino_drawer.publish.manifest import build_meta, upsert_catalog

        meta = build_meta(_make_factsheet(), _IMAGE_URLS)
        catalog = upsert_catalog(None, meta)

        assert catalog["count"] == 1
        assert len(catalog["species"]) == 1
        entry = catalog["species"][0]
        assert entry["slug"] == "tyrannosaurus-rex"
        assert entry["thumb_url"] == _IMAGE_URLS["hero"][400]

    def test_replaces_existing_entry_with_same_slug(self) -> None:
        """upsert_catalog should update (not duplicate) an existing slug."""
        from dino_drawer.publish.manifest import build_meta, upsert_catalog

        meta = build_meta(_make_factsheet(), _IMAGE_URLS)
        catalog = upsert_catalog(None, meta)
        # Upsert same slug again
        catalog2 = upsert_catalog(catalog, meta)

        assert catalog2["count"] == 1

    def test_entries_sorted_by_species_name(self) -> None:
        """Species list in catalog should be sorted alphabetically by species name."""
        from dino_drawer.publish.manifest import build_meta, upsert_catalog

        urls_vel: dict[str, dict[int, str]] = {
            "hero": {400: "https://pub.r2.dev/velociraptor-mongoliensis/hero@400.webp"},
            "skull": {400: "https://pub.r2.dev/velociraptor-mongoliensis/skull@400.webp"},
        }
        meta_trex = build_meta(_make_factsheet("Tyrannosaurus rex"), _IMAGE_URLS)
        meta_vel = build_meta(_make_factsheet("Velociraptor mongoliensis"), urls_vel)

        catalog = upsert_catalog(None, meta_vel)
        catalog = upsert_catalog(catalog, meta_trex)

        names = [s["species"] for s in catalog["species"]]
        assert names == sorted(names)
        assert names[0] == "Tyrannosaurus rex"

    def test_meta_url_derived_from_thumb_url(self) -> None:
        """meta_url should be the sibling meta.json of the thumb URL."""
        from dino_drawer.publish.manifest import build_meta, upsert_catalog

        meta = build_meta(_make_factsheet(), _IMAGE_URLS)
        catalog = upsert_catalog(None, meta)
        entry = catalog["species"][0]

        assert entry["meta_url"].endswith("/meta.json")
        assert entry["meta_url"].startswith("https://pub.r2.dev/tyrannosaurus-rex/")


class TestRemoveFromCatalog:
    """Tests for remove_from_catalog()."""

    def test_removes_matching_slug(self) -> None:
        """remove_from_catalog should strip the entry with the given slug."""
        from dino_drawer.publish.manifest import build_meta, remove_from_catalog, upsert_catalog

        meta = build_meta(_make_factsheet(), _IMAGE_URLS)
        catalog = upsert_catalog(None, meta)
        catalog = remove_from_catalog(catalog, "tyrannosaurus-rex")

        assert catalog["count"] == 0
        assert catalog["species"] == []

    def test_count_decremented(self) -> None:
        """count should reflect the remaining species after removal."""
        from dino_drawer.publish.manifest import build_meta, remove_from_catalog, upsert_catalog

        urls_vel: dict[str, dict[int, str]] = {
            "hero": {400: "https://pub.r2.dev/velociraptor-mongoliensis/hero@400.webp"},
            "skull": {400: "https://pub.r2.dev/velociraptor-mongoliensis/skull@400.webp"},
        }
        meta_trex = build_meta(_make_factsheet("Tyrannosaurus rex"), _IMAGE_URLS)
        meta_vel = build_meta(_make_factsheet("Velociraptor mongoliensis"), urls_vel)
        catalog = upsert_catalog(upsert_catalog(None, meta_trex), meta_vel)
        assert catalog["count"] == 2

        catalog = remove_from_catalog(catalog, "velociraptor-mongoliensis")
        assert catalog["count"] == 1

    def test_nonexistent_slug_is_noop(self) -> None:
        """Removing a slug that is not in the catalog should leave count unchanged."""
        from dino_drawer.publish.manifest import build_meta, remove_from_catalog, upsert_catalog

        meta = build_meta(_make_factsheet(), _IMAGE_URLS)
        catalog = upsert_catalog(None, meta)
        catalog = remove_from_catalog(catalog, "unknown-species")

        assert catalog["count"] == 1
