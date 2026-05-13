"""Tests for dino_drawer.publish.optimize — PNG → WebP conversion."""
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    """Return a fresh temp directory for WebP output."""
    return tmp_path / "webp"


def _make_png(path: Path, width: int, height: int) -> Path:
    """Create a solid-colour PNG at the given size and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    img.save(path, "PNG")
    return path


def test_large_source_produces_three_sizes(tmp_path: Path, tmp_out: Path) -> None:
    """Source 2000×1200 should produce WebP files at 1600, 800 and 400 widths."""
    from dino_drawer.publish.optimize import optimize_image

    src = _make_png(tmp_path / "hero.png", 2000, 1200)
    result = optimize_image(src, tmp_out, "hero")

    assert set(result.keys()) == {1600, 800, 400}
    for w, path in result.items():
        assert path.exists(), f"Missing file for width {w}"
        img = Image.open(path)
        assert img.width == w
        # Verify aspect ratio is preserved (2000/1200 ≈ 1.667)
        expected_h = round(1200 * (w / 2000))
        assert img.height == expected_h, f"Height mismatch for width {w}: {img.height} != {expected_h}"


def test_small_source_skips_larger_widths(tmp_path: Path, tmp_out: Path) -> None:
    """Source 500×500 should only produce 400-width file (skips 1600 and 800)."""
    from dino_drawer.publish.optimize import optimize_image

    src = _make_png(tmp_path / "small.png", 500, 500)
    result = optimize_image(src, tmp_out, "small")

    assert set(result.keys()) == {400}
    assert result[400].exists()


def test_output_files_are_valid_webp(tmp_path: Path, tmp_out: Path) -> None:
    """All generated files should be openable as WebP images."""
    from dino_drawer.publish.optimize import optimize_image

    src = _make_png(tmp_path / "test.png", 2000, 1000)
    result = optimize_image(src, tmp_out, "test")

    for w, path in result.items():
        img = Image.open(path)
        assert img.format == "WEBP", f"File at width {w} is not WebP: {img.format}"


def test_exact_boundary_width_is_included(tmp_path: Path, tmp_out: Path) -> None:
    """Source exactly 1600px wide should produce a 1600-width WebP."""
    from dino_drawer.publish.optimize import optimize_image

    src = _make_png(tmp_path / "exact.png", 1600, 900)
    result = optimize_image(src, tmp_out, "exact")

    assert 1600 in result


def test_output_dir_is_created(tmp_path: Path) -> None:
    """optimize_image should create the output directory if it does not exist."""
    from dino_drawer.publish.optimize import optimize_image

    out_dir = tmp_path / "deep" / "nested" / "dir"
    src = _make_png(tmp_path / "hero.png", 2000, 1000)
    optimize_image(src, out_dir, "hero")

    assert out_dir.is_dir()
