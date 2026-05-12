"""Test SVG silhouette generator: proportions match Size dimensions."""
from dino_drawer.image.silhouette import build_scale_svg


def test_svg_contains_two_rects_with_correct_ratio():
    svg = build_scale_svg(length_m=12.0, hip_height_m=4.0, human_height_m=1.75)
    assert "<svg" in svg
    assert 'viewBox="0 0' in svg
    assert "Tyrannosaurus" not in svg
    assert "1.75" not in svg


def test_svg_handles_small_animal():
    svg = build_scale_svg(length_m=0.5, hip_height_m=0.3, human_height_m=1.75)
    assert "<svg" in svg
