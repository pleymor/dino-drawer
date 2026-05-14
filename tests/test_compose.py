"""Tests for compose.render.screenshot."""
from PIL import Image

from dino_drawer.compose.render import screenshot


def test_screenshot_produces_png_with_expected_dimensions(tmp_path):
    Image.new("RGB", (1024, 640), "olive").save(tmp_path / "hero.png")

    out = screenshot(tmp_path, width=2000, height=1200)
    assert out.exists()
    img = Image.open(out)
    assert img.size == (2000, 1200)
