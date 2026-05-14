"""Produce the final image: the hero photograph centered on a black canvas.

No text, no annotations, no HTML rendering — just resize the Gemini hero PNG
and paste it on a canvas at the target output dimensions.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def screenshot(
    out_dir: Path,
    *,
    width: int = 2000,
    height: int = 1200,
) -> Path:
    """Produce ``final.png`` from ``hero.png``.

    Resizes the Gemini-generated hero while preserving aspect ratio, then
    centers it on a black canvas of *width* × *height*.

    Args:
        out_dir: Directory containing ``hero.png``. ``final.png`` is written here.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        Path to the written ``final.png``.
    """
    out_dir = Path(out_dir)
    src = out_dir / "hero.png"
    dst = (out_dir / "final.png").resolve()

    hero = Image.open(src).convert("RGB")
    src_w, src_h = hero.size
    scale = min(width / src_w, height / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = hero.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (width, height), "#000000")
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    canvas.save(dst, "PNG")
    return dst
