"""Standalone runner: `python -m dino_drawer.research.papers 'Tyrannosaurus rex'`."""
import asyncio
import json
import sys
from pathlib import Path

from .aggregator import fetch_all


async def _run(species: str, out_dir: Path) -> Path:
    """Fetch all paper sources and write results to a JSON file."""
    result = await fetch_all(species)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "papers.json"
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print(f"Wrote {path} ({len(result.papers)} papers)")
    return path


def main() -> int:
    """Entry point for the standalone runner."""
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.research.papers '<species>'", file=sys.stderr)
        return 1
    species = sys.argv[1]
    slug = species.lower().replace(" ", "-")
    out = Path("out") / slug
    asyncio.run(_run(species, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
