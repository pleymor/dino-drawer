"""Wikipedia client — fetches the full plain-text article for a species.

Uses MediaWiki's ``action=query&prop=extracts`` endpoint (no ``exintro``) so
we get the entire article body (Description, Anatomy, Size, Ecology…). The
old ``/page/summary`` REST endpoint only returned the lead paragraph (~700
chars), which was too thin to source mandatory facts like dimensions.
"""
import httpx

from dino_drawer.models import WikipediaContext

_API = "https://en.wikipedia.org/w/api.php"
_PAGE_URL = "https://en.wikipedia.org/wiki/{title}"


async def fetch(species: str) -> WikipediaContext | None:
    """Fetch the full Wikipedia article for `species`. Returns None on miss."""
    title = species.replace(" ", "_")
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
        "format": "json",
        "formatversion": "2",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            _API, params=params, headers={"User-Agent": "dino-drawer/0.1"}
        )
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", []) or []
    if not pages or pages[0].get("missing"):
        return None
    page = pages[0]
    extract = page.get("extract", "") or ""
    if not extract.strip():
        return None
    # MediaWiki may follow a redirect; use the canonical title for the URL.
    canonical_title = page.get("title", title).replace(" ", "_")
    return WikipediaContext(
        url=_PAGE_URL.format(title=canonical_title),
        extract=extract,
    )
