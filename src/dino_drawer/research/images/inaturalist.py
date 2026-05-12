"""iNaturalist API client (extant species only)."""
from pathlib import Path

import httpx

from dino_drawer.models import RawImage

_BASE = "https://api.inaturalist.org/v1"


async def fetch(species: str, out_dir: Path, max_results: int = 10) -> list[RawImage]:
    """Fetch high-quality photos. Returns empty for extinct species."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=20.0) as client:
        tx = await client.get(f"{_BASE}/taxa", params={"q": species, "rank": "species"})
        if tx.status_code != 200:
            return []
        taxa = tx.json().get("results", [])
        if not taxa:
            return []
        taxon = taxa[0]
        if taxon.get("extinct"):
            return []
        taxon_id = taxon["id"]
        obs = await client.get(f"{_BASE}/observations", params={
            "taxon_id": taxon_id,
            "photos": "true",
            "quality_grade": "research",
            "per_page": max_results,
        })
        if obs.status_code != 200:
            return []
        results: list[RawImage] = []
        for i, o in enumerate(obs.json().get("results", [])):
            photos = o.get("photos") or []
            if not photos:
                continue
            url = photos[0].get("url", "").replace("/square.", "/large.")
            if not url:
                continue
            ph = await client.get(url)
            if ph.status_code != 200:
                continue
            ext = "." + url.rsplit(".", 1)[-1].lower().split("?")[0]
            if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                ext = ".jpg"
            fname = f"inat_{i}{ext}"
            (out_dir / fname).write_bytes(ph.content)
            results.append(RawImage(
                id=20_000 + i,
                path=f"refs_raw/{fname}",
                source_url=url,
                source="inaturalist",
                credit=o.get("user", {}).get("login", "iNaturalist user"),
                license=photos[0].get("license_code", "Unknown"),
                width=0,
                height=0,
                search_query=species,
            ))
        return results
