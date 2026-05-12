"""Standalone runner: `python -m dino_drawer.vision <out/species/>`."""
import sys
from pathlib import Path

from .classifier import classify_and_select


def main() -> int:
    """Entry point for the vision classifier CLI.

    Returns:
        Exit code (0 on success, 1 on usage error).
    """
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.vision <species-out-dir>", file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    species = (out_dir.name.replace("-", " ")).title()
    refs = classify_and_select(species=species, out_dir=out_dir)
    print(f"Kept {len(refs.body)} body refs and {len(refs.skull)} skull refs (rejected {refs.rejected_count})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
