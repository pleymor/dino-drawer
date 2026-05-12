"""Tests for HTML rendering and Playwright screenshot."""
import shutil
from pathlib import Path

from PIL import Image

from dino_drawer.models import FactSheet, Annotation, SkullView, Size, Reference, VisualReferences
from dino_drawer.compose.render import render_html, screenshot


def _basic_factsheet() -> FactSheet:
    return FactSheet(
        species="Tyrannosaurus rex",
        subtitle="Reconstitution basée sur les données 2020-2024",
        annotations=[
            Annotation(region="tête", facts=["Museau robuste"], source_ids=["wikipedia"]),
            Annotation(region="peau_et_couverture", facts=["Peau écailleuse"], source_ids=["wikipedia"]),
            Annotation(region="membres_anterieurs", facts=["Courts"], source_ids=["wikipedia"]),
            Annotation(region="membres_posterieurs", facts=["Puissants"], source_ids=["wikipedia"]),
            Annotation(region="queue", facts=["Longue"], source_ids=["wikipedia"]),
        ],
        skull_view=SkullView(facts=["Mâchoire massive"], scale_cm=50, source_ids=["wikipedia"]),
        size=Size(length_m=[12, 13], hip_height_m=[3.5, 4], source_ids=["wikipedia"]),
        conclusion="T. rex avait probablement des lèvres.",
        references=[Reference(id="wikipedia", citation_short="Wikipedia", doi=None, title="T. rex")],
        visual_references=VisualReferences(body=[], skull=[]),
        image_prompt="...",
    )


def test_render_html_contains_all_sections(tmp_path):
    Image.new("RGB", (1024, 640), "olive").save(tmp_path / "hero.png")
    Image.new("RGB", (640, 480), "gray").save(tmp_path / "skull.png")
    (tmp_path / "silhouette.svg").write_text("<svg></svg>")

    html = render_html(_basic_factsheet(), out_dir=tmp_path)
    assert "Tyrannosaurus rex" in html
    assert "Museau robuste" in html
    assert "T. rex avait probablement des lèvres" in html
    for region in ("tête", "peau", "membres", "queue"):
        assert region.lower() in html.lower()


def test_screenshot_produces_png_with_expected_dimensions(tmp_path):
    Image.new("RGB", (1024, 640), "olive").save(tmp_path / "hero.png")
    Image.new("RGB", (640, 480), "gray").save(tmp_path / "skull.png")
    (tmp_path / "silhouette.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100"/></svg>')

    out = screenshot(_basic_factsheet(), out_dir=tmp_path, width=2000, height=1200)
    assert out.exists()
    img = Image.open(out)
    assert img.size == (2000, 1200)
