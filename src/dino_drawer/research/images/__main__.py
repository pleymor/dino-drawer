"""Standalone runner: `python -m dino_drawer.research.images 'Tyrannosaurus rex'`."""
import asyncio
import sys
from pathlib import Path

from .aggregator import fetch_all


def main() -> int:
    """Entry point for the standalone image research runner."""
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.research.images '<species>'", file=sys.stderr)
        return 1
    species = sys.argv[1]
    slug = species.lower().replace(" ", "-")
    result = asyncio.run(fetch_all(species, Path("out") / slug))
    print(f"Wrote {len(result.images)} raw images to out/{slug}/refs_raw/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
