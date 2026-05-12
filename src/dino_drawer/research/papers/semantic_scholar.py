"""Semantic Scholar API client. Returns recent papers about the species."""
import httpx

from dino_drawer.models import Paper

_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"


async def fetch(species: str, limit: int = 20, since_year: int = 2020) -> list[Paper]:
    """Search recent papers mentioning the species. Returns empty list on error."""
    params = {
        "query": species,
        "limit": limit,
        "year": f"{since_year}-",
        "fields": "title,abstract,year,authors,externalIds",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(_BASE, params=params, headers={"User-Agent": "dino-drawer/0.1"})
    if r.status_code != 200:
        return []
    out: list[Paper] = []
    for item in r.json().get("data", []):
        if not item.get("abstract"):
            continue
        out.append(Paper(
            doi=(item.get("externalIds") or {}).get("DOI"),
            title=item["title"],
            authors=[a["name"] for a in item.get("authors", [])],
            year=item.get("year"),
            abstract=item["abstract"],
            source="semantic_scholar",
        ))
    return out
