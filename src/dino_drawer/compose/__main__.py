"""Standalone runner: ``python -m dino_drawer.compose <out/species/>``.

Reads ``hero.png`` from the given species output directory, produces ``final.png``,
and prints the output path.
"""
import sys
from pathlib import Path

from .render import screenshot


def main() -> int:
    """Entry point for the compose module CLI.

    Returns:
        Exit code: 0 on success, 1 if arguments are missing.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.compose <species-out-dir>", file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    out = screenshot(out_dir)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
