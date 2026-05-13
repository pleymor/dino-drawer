"""Convert master PNGs to web-optimised WebP at multiple resolutions."""
from pathlib import Path

from PIL import Image

_WIDTHS = (1600, 800, 400)


def optimize_image(src_png: Path, out_dir: Path, basename: str) -> dict[int, Path]:
    """Resize a PNG into ``<basename>@<width>.webp`` files in *out_dir*.

    Resizes preserving aspect ratio.  Skips a width if the source is smaller
    than that width (no upscaling).  Returns a mapping ``width → output path``
    for every size produced.

    Parameters
    ----------
    src_png:
        Path to the source PNG file.
    out_dir:
        Directory where the WebP files will be written (created if missing).
    basename:
        Stem used in the output filename, e.g. ``"hero"`` → ``hero@1600.webp``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_png).convert("RGB")
    src_w, src_h = img.size
    out: dict[int, Path] = {}
    for w in _WIDTHS:
        if w > src_w:
            continue
        scale = w / src_w
        new_h = round(src_h * scale)
        resized = img.resize((w, new_h), Image.LANCZOS)
        path = out_dir / f"{basename}@{w}.webp"
        resized.save(path, "WEBP", quality=85, method=6)
        out[w] = path
    return out
