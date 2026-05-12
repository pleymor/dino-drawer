"""Tests for the three paper sources and the aggregator."""
import json
from pathlib import Path

import httpx
import pytest
import respx

from dino_drawer.research.papers import wikipedia, semantic_scholar, openalex
from dino_drawer.research.papers import fetch_all
from dino_drawer.models import PapersFile

FIXTURES = Path(__file__).parent / "fixtures"


@respx.mock
async def test_wikipedia_returns_extract():
    payload = json.loads((FIXTURES / "wikipedia_trex.json").read_text())
    respx.get(url__regex=r"https://en\.wikipedia\.org/api/rest_v1/page/summary/.+").mock(
        return_value=httpx.Response(200, json=payload)
    )
    ctx = await wikipedia.fetch("Tyrannosaurus rex")
    assert "Tyrannosaurus" in ctx.extract
    assert ctx.url.startswith("https://en.wikipedia.org/")


@respx.mock
async def test_semantic_scholar_returns_papers():
    payload = json.loads((FIXTURES / "semantic_scholar_trex.json").read_text())
    respx.get(url__regex=r"https://api\.semanticscholar\.org/.+").mock(
        return_value=httpx.Response(200, json=payload)
    )
    papers = await semantic_scholar.fetch("Tyrannosaurus rex")
    assert len(papers) == 1
    assert papers[0].doi == "10.1126/science.abc1234"
    assert papers[0].year == 2023


@respx.mock
async def test_openalex_returns_papers():
    payload = json.loads((FIXTURES / "openalex_trex.json").read_text())
    respx.get(url__regex=r"https://api\.openalex\.org/.+").mock(
        return_value=httpx.Response(200, json=payload)
    )
    papers = await openalex.fetch("Tyrannosaurus rex")
    assert len(papers) == 1
    assert papers[0].doi == "10.1098/rspb.2024.0001"
    assert "skin" in papers[0].abstract.lower()


@respx.mock
async def test_fetch_all_dedupes_by_doi():
    wiki_payload = json.loads((FIXTURES / "wikipedia_trex.json").read_text())
    ss_payload = json.loads((FIXTURES / "semantic_scholar_trex.json").read_text())
    oa_payload = json.loads((FIXTURES / "openalex_trex.json").read_text())
    # Make openalex return same DOI as semantic scholar to test dedup
    oa_payload["results"][0]["doi"] = "https://doi.org/10.1126/science.abc1234"
    respx.get(url__regex=r"https://en\.wikipedia\.org/.+").mock(return_value=httpx.Response(200, json=wiki_payload))
    respx.get(url__regex=r"https://api\.semanticscholar\.org/.+").mock(return_value=httpx.Response(200, json=ss_payload))
    respx.get(url__regex=r"https://api\.openalex\.org/.+").mock(return_value=httpx.Response(200, json=oa_payload))

    result: PapersFile = await fetch_all("Tyrannosaurus rex")
    assert len({p.doi for p in result.papers if p.doi}) == 1
    assert result.wikipedia is not None


@respx.mock
async def test_no_papers_found_still_returns_wikipedia():
    wiki_payload = json.loads((FIXTURES / "wikipedia_trex.json").read_text())
    respx.get(url__regex=r"https://en\.wikipedia\.org/.+").mock(return_value=httpx.Response(200, json=wiki_payload))
    respx.get(url__regex=r"https://api\.semanticscholar\.org/.+").mock(return_value=httpx.Response(200, json={"total": 0, "data": []}))
    respx.get(url__regex=r"https://api\.openalex\.org/.+").mock(return_value=httpx.Response(200, json={"results": []}))

    result = await fetch_all("Obscurus obscurus")
    assert result.papers == []
    assert result.wikipedia is not None
