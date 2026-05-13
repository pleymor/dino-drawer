"""Render the final image: just the hero photograph at the target resolution.

The factsheet text and visual references stay in ``factsheet.json`` for any
downstream consumer; the rendered output is a clean PNG with no annotations,
panels, or text overlays.
"""
from __future__ import annotations

import base64
import shutil
from pathlib import Path

from jinja2 import Template
from PIL import Image

from dino_drawer.models import FactSheet

_REGION_TO_CLASS: dict[str, str] = {
    "tête": "tete",
    "peau_et_couverture": "peau",
    "membres_anterieurs": "bras",
    "membres_posterieurs": "pattes",
    "queue": "queue",
}

_TEMPLATES = Path(__file__).parent / "templates"


def _data_uri(path: Path) -> str:
    """Convert a PNG path to a base64 data URI for inline embedding.

    Args:
        path: Path to the PNG image file.

    Returns:
        A ``data:image/png;base64,...`` string suitable for use in an ``<img src>``.
    """
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def render_html(fs: FactSheet, out_dir: Path, lang: str = "fr") -> str:
    """Render the infographic HTML string with inlined images.

    Reads ``hero.png``, ``skull.png``, and ``silhouette.svg`` from *out_dir*,
    embeds them as data URIs / inline SVG, and returns the complete HTML.

    Args:
        fs: The :class:`~dino_drawer.models.FactSheet` containing all content.
        out_dir: Directory containing ``hero.png``, ``skull.png``, and ``silhouette.svg``.
        lang: BCP-47 language tag for the ``<html lang>`` attribute (default ``"fr"``).

    Returns:
        Rendered HTML string.
    """
    css = (_TEMPLATES / "infographic.css").read_text()
    tmpl = Template((_TEMPLATES / "infographic.html").read_text())
    return tmpl.render(
        fs=fs,
        css=css,
        lang=lang,
        hero_data_uri=_data_uri(out_dir / "hero.png"),
        skull_data_uri=_data_uri(out_dir / "skull.png"),
        silhouette_svg=(out_dir / "silhouette.svg").read_text(),
        region_class=lambda r: _REGION_TO_CLASS.get(r, ""),
    )


def screenshot(
    fs: FactSheet,
    out_dir: Path,
    *,
    width: int = 2000,
    height: int = 1200,
) -> Path:
    """Produce ``final.png`` from the hero image, with no text or panels.

    Resizes the Gemini-generated hero to the target width while preserving
    aspect ratio, then centers it on a black canvas of *width* × *height* if
    the aspect ratios differ. No text, no annotations, no overlays.

    Args:
        fs: The factsheet (kept in signature for back-compat; not rendered).
        out_dir: Directory containing ``hero.png``. ``final.png`` is written here.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        Path to the written ``final.png``.
    """
    del fs  # factsheet text no longer rendered into the image
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


def render_legacy_html(fs: FactSheet, out_dir: Path, lang: str = "fr") -> str:
    """Render the old annotated-infographic HTML.

    Kept for callers that still want the annotated layout. Not used by the
    default pipeline anymore (see :func:`screenshot`).
    """
    return render_html(fs, out_dir, lang=lang)
