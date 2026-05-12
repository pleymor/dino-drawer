"""OpenAlex API client. Complement to Semantic Scholar."""
import httpx

from dino_drawer.models import Paper

_BASE = "https://api.openalex.org/works"


def _decode_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


async def fetch(species: str, limit: int = 20, since_year: int = 2020) -> list[Paper]:
    """Search OpenAlex for works mentioning the species."""
    params = {
        "search": species,
        "per-page": limit,
        "filter": f"from_publication_date:{since_year}-01-01",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(_BASE, params=params, headers={"User-Agent": "dino-drawer/0.1"})
    if r.status_code != 200:
        return []
    out: list[Paper] = []
    for item in r.json().get("results", []):
        abstract = _decode_abstract(item.get("abstract_inverted_index"))
        if not abstract:
            continue
        doi = item.get("doi", "")
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        out.append(Paper(
            doi=doi or None,
            title=item.get("title", ""),
            authors=[a["author"]["display_name"] for a in item.get("authorships", [])],
            year=item.get("publication_year"),
            abstract=abstract,
            source="openalex",
        ))
    return out
