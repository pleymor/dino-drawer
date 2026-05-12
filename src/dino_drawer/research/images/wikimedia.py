"""Wikimedia Commons image scraper for a species, with license metadata."""
import asyncio
from pathlib import Path

import httpx

from dino_drawer.models import RawImage

_API = "https://commons.wikimedia.org/w/api.php"
_REJECT_KEYWORDS = ("chart", "phylogeny", "cladogram", "range_map", "rangemap", "graph")
_MIN_W, _MIN_H = 800, 600
_ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".webp")
_RETRY_DELAY = 2.0  # seconds to wait on 429
# Wikimedia requires a descriptive User-Agent identifying the tool and contact
_UA = "DinoDrawer/0.1 (https://github.com/user/dino-drawer; paleontology-research-tool) python-httpx/0.27"


def _rejected_by_name(filename: str) -> bool:
    """Reject filenames containing chart-like keywords before any download."""
    low = filename.lower()
    if not any(low.endswith(ext) for ext in _ALLOWED_EXT):
        return True
    return any(k in low for k in _REJECT_KEYWORDS)


async def _search_titles(client: httpx.AsyncClient, query: str, limit: int) -> list[str]:
    """Search Wikimedia Commons for file titles matching the query. Retries on 429."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srnamespace": 6,
        "srsearch": query,
        "srlimit": limit,
    }
    for attempt in range(3):
        r = await client.get(_API, params=params, headers={"User-Agent": _UA})
        if r.status_code == 429:
            await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
            continue
        r.raise_for_status()
        return [hit["title"] for hit in r.json().get("query", {}).get("search", [])]
    return []


async def _imageinfo(client: httpx.AsyncClient, title: str) -> dict | None:
    """Fetch image info including URL, dimensions, and license metadata. Retries on 429."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
    }
    for attempt in range(3):
        r = await client.get(_API, params=params, headers={"User-Agent": _UA})
        if r.status_code == 429:
            await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
            continue
        if r.status_code != 200:
            return None
        pages = r.json().get("query", {}).get("pages", {})
        for _, page in pages.items():
            infos = page.get("imageinfo")
            if infos:
                return {**infos[0], "title": page.get("title", title)}
        return None
    return None


async def fetch(species: str, out_dir: Path, max_results: int = 30) -> list[RawImage]:
    """Search Wikimedia Commons, download images, return metadata of survivors."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[RawImage] = []
    queries = [f"{species} paleoart", species]
    async with httpx.AsyncClient(timeout=30.0) as client:
        seen_titles: set[str] = set()
        next_id = 0
        for q in queries:
            titles = await _search_titles(client, q, max_results)
            for title in titles:
                if title in seen_titles or _rejected_by_name(title):
                    continue
                seen_titles.add(title)
                await asyncio.sleep(0.3)  # avoid 429 from Wikimedia
                info = await _imageinfo(client, title)
                if not info:
                    continue
                if info.get("width", 0) < _MIN_W or info.get("height", 0) < _MIN_H:
                    continue
                url = info.get("url")
                if not url:
                    continue
                # download with retry on 429
                resp = None
                for attempt in range(3):
                    resp = await client.get(url, headers={"User-Agent": _UA})
                    if resp.status_code == 429:
                        await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
                        continue
                    break
                if resp is None or resp.status_code != 200:
                    continue
                ext = "." + url.rsplit(".", 1)[-1].lower().split("?")[0]
                if ext not in _ALLOWED_EXT:
                    ext = ".jpg"
                fname = f"wm_{next_id}{ext}"
                (out_dir / fname).write_bytes(resp.content)
                meta = info.get("extmetadata", {})
                results.append(RawImage(
                    id=next_id,
                    path=f"refs_raw/{fname}",
                    source_url=url,
                    source="wikimedia_commons",
                    credit=meta.get("Artist", {}).get("value", "Unknown"),
                    license=meta.get("LicenseShortName", {}).get("value", "Unknown"),
                    width=info["width"],
                    height=info["height"],
                    search_query=q,
                ))
                next_id += 1
                if len(results) >= max_results:
                    return results
    return results
