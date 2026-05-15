"""Standalone runner: `python -m dino_drawer.video <out/species/>`."""
from __future__ import annotations

import sys
from pathlib import Path

from dino_drawer.models import FactSheet
from .generate import generate_video


def main() -> int:
    """Read factsheet + hero.png from *out_dir* and write hero.mp4 next to them."""
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.video <species-out-dir>", file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    fs = FactSheet.model_validate_json((out_dir / "factsheet.json").read_text())
    mp4 = generate_video(factsheet=fs, out_dir=out_dir)
    print(f"Wrote {mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
