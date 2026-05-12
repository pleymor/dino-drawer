"""Standalone runner: `python -m dino_drawer.synthesis <out/species/>`."""
import sys
from pathlib import Path

from .main import synthesize


def main() -> int:
    """
    Entry point for running synthesis from the command line.

    Expects a single positional argument: the species output directory.
    The species name is inferred from the directory name by replacing
    hyphens with spaces and applying title case.

    Returns:
        Exit code: 0 on success, 1 if the output directory argument is missing.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.synthesis <species-out-dir>", file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    species = out_dir.name.replace("-", " ").title()
    fs = synthesize(species=species, out_dir=out_dir)
    print(
        f"Wrote {out_dir}/factsheet.json — "
        f"{len(fs.annotations)} regions, {len(fs.references)} refs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
