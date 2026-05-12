"""Combine paper sources, dedupe by DOI, attach Wikipedia context."""
import asyncio

from dino_drawer.models import PapersFile, Paper
from . import wikipedia, semantic_scholar, openalex


async def fetch_all(species: str) -> PapersFile:
    """Run all three sources concurrently and dedupe by DOI."""
    wiki, ss, oa = await asyncio.gather(
        wikipedia.fetch(species),
        semantic_scholar.fetch(species),
        openalex.fetch(species),
        return_exceptions=False,
    )
    seen: set[str] = set()
    merged: list[Paper] = []
    for p in (*ss, *oa):
        key = p.doi or f"{p.title}|{p.year}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)
    return PapersFile(species=species, wikipedia=wiki, papers=merged)
