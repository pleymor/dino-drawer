"""Generate a scale comparison SVG: animal silhouette next to a human."""
from __future__ import annotations


def build_scale_svg(
    length_m: float,
    hip_height_m: float,
    human_height_m: float = 1.75,
    *,
    width_px: int = 600,
    height_px: int = 200,
) -> str:
    """Build an SVG comparing the animal silhouette (rectangle proxy) to a human.

    The animal is drawn as a horizontal rounded rect (length × hip_height),
    the human as a tall thin rect of human_height_m.
    Both are scaled to fit in `width_px × height_px`.

    Args:
        length_m: Animal length in meters.
        hip_height_m: Animal hip height in meters.
        human_height_m: Human height in meters (default 1.75m).
        width_px: SVG canvas width in pixels (default 600).
        height_px: SVG canvas height in pixels (default 200).

    Returns:
        SVG string with scale comparison silhouettes.
    """
    margin = 20
    total_world_w = length_m + 1.5
    scale = (width_px - 2 * margin) / total_world_w
    animal_w = length_m * scale
    animal_h = hip_height_m * scale
    human_h = human_height_m * scale
    human_w = 0.4 * scale

    baseline_y = height_px - margin
    animal_y = baseline_y - animal_h
    human_y = baseline_y - human_h
    animal_x = margin
    human_x = margin + animal_w + 0.5 * scale

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}">
  <rect x="{animal_x:.1f}" y="{animal_y:.1f}" width="{animal_w:.1f}" height="{animal_h:.1f}" rx="6" fill="#222"/>
  <rect x="{human_x:.1f}" y="{human_y:.1f}" width="{human_w:.1f}" height="{human_h:.1f}" rx="3" fill="#888"/>
  <line x1="{margin}" y1="{baseline_y}" x2="{width_px - margin}" y2="{baseline_y}" stroke="#444" stroke-width="1"/>
</svg>"""
