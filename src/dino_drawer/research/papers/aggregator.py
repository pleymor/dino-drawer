"""Combine paper sources, dedupe by DOI, attach Wikipedia context.

Each source is fetched independently; transient failures (timeouts, 5xx) of
one source must NOT kill the whole pipeline. We log a one-line warning and
keep going with whatever did succeed.
"""
import asyncio
import sys

from dino_drawer.models import PapersFile, Paper
from . import wikipedia, semantic_scholar, openalex


async def fetch_all(species: str) -> PapersFile:
    """Run all three sources concurrently. Source failures are logged and skipped."""
    wiki_r, ss_r, oa_r = await asyncio.gather(
        wikipedia.fetch(species),
        semantic_scholar.fetch(species),
        openalex.fetch(species),
        return_exceptions=True,
    )

    def _unwrap_list(name: str, r: object) -> list[Paper]:
        if isinstance(r, BaseException):
            print(f"[papers] {name} failed: {type(r).__name__}: {r}", file=sys.stderr, flush=True)
            return []
        return r  # type: ignore[return-value]

    wiki = None
    if isinstance(wiki_r, BaseException):
        print(f"[papers] wikipedia failed: {type(wiki_r).__name__}: {wiki_r}", file=sys.stderr, flush=True)
    else:
        wiki = wiki_r

    ss = _unwrap_list("semantic_scholar", ss_r)
    oa = _unwrap_list("openalex", oa_r)

    seen: set[str] = set()
    merged: list[Paper] = []
    for p in (*ss, *oa):
        key = p.doi or f"{p.title}|{p.year}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)
    return PapersFile(species=species, wikipedia=wiki, papers=merged)
