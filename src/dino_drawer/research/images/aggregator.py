"""Run all image sources concurrently, write refs_raw.json."""
import asyncio
import json
from pathlib import Path

from dino_drawer.models import RawImagesFile
from . import wikimedia, phylopic, inaturalist


async def fetch_all(species: str, out_dir: Path, max_total: int = 50) -> RawImagesFile:
    """Run all 3 sources in parallel, write refs_raw.json."""
    raw_dir = Path(out_dir) / "refs_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Reserve slots for PhyloPic (3) and iNaturalist (10) so Wikimedia doesn't crowd them out
    wm_limit = max(1, max_total - 3 - 10)
    wm, pp, inat = await asyncio.gather(
        wikimedia.fetch(species, raw_dir, max_results=wm_limit),
        phylopic.fetch(species, raw_dir, max_results=3),
        inaturalist.fetch(species, raw_dir, max_results=10),
        return_exceptions=False,
    )
    all_images = [*wm, *pp, *inat][:max_total]
    result = RawImagesFile(species=species, images=all_images)
    (Path(out_dir) / "refs_raw.json").write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    )
    return result
