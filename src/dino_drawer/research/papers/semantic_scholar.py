"""Semantic Scholar API client. Returns recent papers about the species."""
import asyncio

import httpx

from dino_drawer.models import Paper

_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

# The Semantic Scholar public API frequently throttles and times out; tolerate
# transient failures with two retries on network errors / 5xx.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0


async def fetch(species: str, limit: int = 20, since_year: int = 2020) -> list[Paper]:
    """Search recent papers mentioning the species. Returns empty list on persistent error."""
    params = {
        "query": species,
        "limit": limit,
        "year": f"{since_year}-",
        "fields": "title,abstract,year,authors,externalIds",
    }
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(
                    _BASE, params=params, headers={"User-Agent": "dino-drawer/0.1"}
                )
            if r.status_code == 200:
                break
            if 500 <= r.status_code < 600 or r.status_code == 429:
                last_exc = httpx.HTTPStatusError(
                    f"S2 returned {r.status_code}", request=r.request, response=r
                )
            else:
                return []  # 4xx (other than 429) — don't retry
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
        # Backoff before the next attempt.
        if attempt < _MAX_ATTEMPTS - 1:
            await asyncio.sleep(_RETRY_BACKOFF_S * (attempt + 1))
    else:
        # All attempts failed.
        if last_exc is not None:
            raise last_exc
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
