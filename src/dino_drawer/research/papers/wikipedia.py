"""Wikipedia REST API client for the species article extract."""
import httpx

from dino_drawer.models import WikipediaContext

_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"


async def fetch(species: str) -> WikipediaContext | None:
    """Fetch the Wikipedia summary for `species`. Returns None on 404."""
    url = f"{_BASE}/{species.replace(' ', '_')}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, headers={"User-Agent": "dino-drawer/0.1"})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    return WikipediaContext(
        url=data.get("content_urls", {}).get("desktop", {}).get("page", url),
        extract=data.get("extract", ""),
    )
