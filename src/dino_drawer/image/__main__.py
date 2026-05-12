"""Standalone runner: `python -m dino_drawer.image <out/species/>`."""
import sys
from pathlib import Path

from dino_drawer.models import FactSheet
from .diffusion import generate_assets


def main() -> int:
    """Entry point for the image generation module.

    Reads ``factsheet.json`` from the given species output directory,
    runs the full diffusion pipeline, and writes ``hero.png``,
    ``skull.png``, and ``silhouette.svg`` to the same directory.

    Returns:
        0 on success, 1 if no output directory was provided.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.image <species-out-dir>", file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    fs = FactSheet.model_validate_json((out_dir / "factsheet.json").read_text())
    generate_assets(factsheet=fs, out_dir=out_dir)
    print(f"Wrote {out_dir}/hero.png, skull.png, silhouette.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
