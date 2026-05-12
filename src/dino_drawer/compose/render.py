"""Render the infographic HTML and screenshot it with Playwright."""
from __future__ import annotations

import base64
from pathlib import Path

from jinja2 import Template

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
    """Render HTML and use Playwright to screenshot it to ``final.png``.

    Writes an intermediate ``_infographic.html`` file to *out_dir*, then
    opens it in a headless Chromium browser and captures a full-viewport
    screenshot at the requested dimensions.

    Args:
        fs: The :class:`~dino_drawer.models.FactSheet` containing all content.
        out_dir: Directory containing source assets and where ``final.png`` is written.
        width: Viewport and output image width in pixels (default 2000).
        height: Viewport and output image height in pixels (default 1200).

    Returns:
        Path to the written ``final.png`` file.
    """
    from playwright.sync_api import sync_playwright

    html = render_html(fs, out_dir)
    html_path = out_dir / "_infographic.html"
    html_path.write_text(html)

    out = out_dir / "final.png"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(html_path.as_uri())
        page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": width, "height": height})
        browser.close()
    return out
