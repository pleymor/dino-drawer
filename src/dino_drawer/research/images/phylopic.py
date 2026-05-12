"""PhyloPic API v2 client for vector silhouettes. Returns raster downloads.

Resolution strategy:
1. Query GBIF to get the taxon ID for the species
2. Use PhyloPic resolve endpoint with the GBIF ID to find the node UUID
3. Fetch clade images and download the largest rasters
"""
from pathlib import Path

import httpx

from dino_drawer.models import RawImage

_PHYLOPIC = "https://api.phylopic.org"
_GBIF = "https://api.gbif.org/v1"


async def _gbif_taxon_key(client: httpx.AsyncClient, species: str) -> int | None:
    """Return the GBIF taxon key for a species name, or None if not found."""
    r = await client.get(f"{_GBIF}/species/match", params={"name": species, "rank": "SPECIES"})
    if r.status_code != 200:
        return None
    data = r.json()
    return data.get("usageKey") or data.get("speciesKey")


async def _phylopic_node_via_gbif(
    client: httpx.AsyncClient, gbif_key: int, build: int
) -> dict | None:
    """Resolve a PhyloPic node from a GBIF taxon key."""
    r = await client.get(
        f"{_PHYLOPIC}/resolve/gbif.org/species/{gbif_key}",
        params={"build": build},
    )
    if r.status_code != 200:
        return None
    return r.json()


async def fetch(species: str, out_dir: Path, max_results: int = 3) -> list[RawImage]:
    """Fetch silhouettes from PhyloPic v2. Resolves node via GBIF ID."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        root_r = await client.get(f"{_PHYLOPIC}/")
        if root_r.status_code != 200:
            return []
        build = root_r.json().get("build")
        if not build:
            return []

        gbif_key = await _gbif_taxon_key(client, species)
        if not gbif_key:
            return []

        node = await _phylopic_node_via_gbif(client, gbif_key, build)
        if not node:
            return []

        clade_href = node.get("_links", {}).get("cladeImages", {}).get("href")
        if not clade_href:
            return []

        # Extract filter_clade UUID from href (avoids double-encoding query params)
        clade_uuid = None
        for part in clade_href.split("&"):
            if part.startswith("filter_clade="):
                clade_uuid = part.split("=", 1)[1]
                break
        if not clade_uuid:
            return []

        r2 = await client.get(
            f"{_PHYLOPIC}/images",
            params={
                "build": build,
                "filter_clade": clade_uuid,
                "embed_items": "true",
                "page": 0,
            },
        )
        if r2.status_code != 200:
            return []

        image_items = r2.json().get("_embedded", {}).get("items", [])[:max_results]
        results: list[RawImage] = []
        for i, item in enumerate(image_items):
            raster_files = item.get("_links", {}).get("rasterFiles", [])
            if not raster_files:
                continue
            src_url = raster_files[0].get("href")
            if not src_url:
                continue
            raster = await client.get(src_url)
            if raster.status_code != 200:
                continue
            fname = f"pp_{i}.png"
            (out_dir / fname).write_bytes(raster.content)
            sizes_str = raster_files[0].get("sizes", "0x0")
            try:
                w, h = (int(x) for x in sizes_str.split("x"))
            except ValueError:
                w, h = 0, 0
            license_href = item.get("_links", {}).get("license", {}).get("href", "CC0")
            results.append(RawImage(
                id=10_000 + i,
                path=f"refs_raw/{fname}",
                source_url=src_url,
                source="phylopic",
                credit=item.get("attribution") or "PhyloPic contributor",
                license=license_href,
                width=w,
                height=h,
                search_query=species,
            ))
        return results
