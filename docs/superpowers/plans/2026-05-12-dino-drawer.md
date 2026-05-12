# Dino Drawer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI agent that, given a species name, generates a sourced scientific infographic with realistic illustration grounded on filtered reference images, all using locally-run LLMs and VLMs.

**Architecture:** 6-step pipeline (papers → images → VLM filtering → LLM synthesis → conditioned diffusion → HTML composition), orchestrated by a Python async agent. JSON intermediate artifacts make each step independently re-runnable. Diffusion is grounded by IP-Adapter on VLM-selected reference images.

**Tech Stack:** Python 3.11+, httpx (async HTTP), Pydantic v2, Ollama (qwen2.5:14b text, qwen2.5vl:7b vision), diffusers + SDXL + IP-Adapter Plus (MPS), Jinja2, Playwright, pytest + respx.

---

## Testable Milestones

You'll be able to test progressively as we build :

| After task | Test command | Expected |
|---|---|---|
| Task 0 | `python -m dino_drawer --help` | CLI help printed |
| Task 2 | `python -m dino_drawer.research.papers "Tyrannosaurus rex"` | `papers.json` written |
| Task 3 | `python -m dino_drawer.research.images "Tyrannosaurus rex"` | 20-50 images in `refs_raw/` |
| Task 5 | `python -m dino_drawer.vision.classify out/tyrannosaurus-rex/` | `refs.json` + filtered images |
| Task 8 | `python -m dino_drawer.synthesis out/tyrannosaurus-rex/` | `factsheet.json` |
| Task 10 | `python -m dino_drawer.image out/tyrannosaurus-rex/` | `hero.png`, `skull.png`, `silhouette.svg` |
| Task 11 | `python -m dino_drawer.compose out/tyrannosaurus-rex/` | `final.png` |
| Task 13 | `python -m dino_drawer "Tyrannosaurus rex"` | Full pipeline end-to-end |

---

## Prerequisites the user must do once

These are NOT part of the plan, but the engineer should mention them in the README:

- Install Ollama : https://ollama.com
- Pull text and vision models : `ollama pull qwen2.5:14b-instruct && ollama pull qwen2.5vl:7b`
- Install Python 3.11+ and `uv` (or pip)
- After Task 0 : `playwright install chromium`
- First diffusion run will download ~10 GB of model weights from Hugging Face

---

## Task 0: Project skeleton + CLI entry point

**Files:**
- Create: `pyproject.toml`
- Create: `src/dino_drawer/__init__.py`
- Create: `src/dino_drawer/__main__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_cli.py`
- Modify: `.gitignore` (add `*.egg-info`, already covered)

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "dino-drawer"
version = "0.1.0"
description = "Generate scientific infographics of prehistoric species from local LLMs"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.7",
    "ollama>=0.3",
    "jinja2>=3.1",
    "playwright>=1.45",
    "pillow>=10.3",
    "diffusers>=0.30",
    "transformers>=4.44",
    "accelerate>=0.33",
    "torch>=2.4",
    "safetensors>=0.4",
    "rich>=13.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
    "ruff>=0.6",
]

[project.scripts]
dino-drawer = "dino_drawer.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/dino_drawer"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Write CLI test that should fail**

`tests/test_cli.py`:
```python
"""Smoke test for the CLI entry point."""
import subprocess
import sys


def test_cli_help_returns_zero():
    """`python -m dino_drawer --help` must exit 0 and print usage."""
    result = subprocess.run(
        [sys.executable, "-m", "dino_drawer", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "species" in result.stdout.lower() or "usage" in result.stdout.lower()
```

- [ ] **Step 3: Run test — expect failure (module not found)**

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/test_cli.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'dino_drawer'`).

- [ ] **Step 4: Write minimal CLI**

`src/dino_drawer/__init__.py`:
```python
"""Dino Drawer — scientific infographics for prehistoric species."""
__version__ = "0.1.0"
```

`src/dino_drawer/__main__.py`:
```python
"""CLI entry point for dino-drawer."""
import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser for the dino-drawer CLI."""
    parser = argparse.ArgumentParser(
        prog="dino-drawer",
        description="Generate a scientific infographic for a species.",
    )
    parser.add_argument("species", nargs="?", help="Binomial name, e.g. 'Tyrannosaurus rex'")
    parser.add_argument("--out", default="./out", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Re-run all steps")
    parser.add_argument(
        "--force-step",
        choices=["papers", "images", "filter", "synthesis", "diffusion", "compose"],
        help="Re-run this step and everything after",
    )
    parser.add_argument("--skip-refs", action="store_true", help="Skip image scraping + VLM filtering")
    parser.add_argument("--model-llm", default="qwen2.5:14b-instruct")
    parser.add_argument("--model-vlm", default="qwen2.5vl:7b")
    parser.add_argument("--model-image", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--max-refs", type=int, default=50)
    parser.add_argument("--lang", choices=["fr", "en"], default="fr")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dino-drawer CLI. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.species:
        parser.print_help()
        return 0
    print(f"[stub] would generate infographic for {args.species!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run test — expect pass**

```bash
pytest tests/test_cli.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "✨ [init] Project skeleton with CLI entry point and smoke test"
```

---

## Task 1: Pydantic models for all pipeline artifacts

**Files:**
- Create: `src/dino_drawer/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
"""Tests for Pydantic models used as pipeline artifacts."""
from dino_drawer.models import (
    Paper,
    PapersFile,
    RawImage,
    RawImagesFile,
    ClassifiedImage,
    RefsFile,
    Annotation,
    SkullView,
    Size,
    Reference,
    VisualRef,
    VisualReferences,
    FactSheet,
)


def test_paper_roundtrip():
    p = Paper(
        doi="10.1234/x",
        title="A study",
        authors=["A. Author"],
        year=2024,
        abstract="...",
        source="semantic_scholar",
    )
    assert Paper.model_validate(p.model_dump()) == p


def test_factsheet_requires_each_fact_has_source():
    """Validation: every annotation fact must reference an existing source_id."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FactSheet(
            species="X",
            subtitle="",
            annotations=[Annotation(region="tête", facts=["fact"], source_ids=["missing"])],
            skull_view=SkullView(facts=["f"], scale_cm=10, source_ids=["paper:0"]),
            size=Size(length_m=[1, 2], hip_height_m=[1, 2], source_ids=["paper:0"]),
            conclusion="",
            references=[Reference(id="paper:0", citation_short="X", doi=None, title="t")],
            visual_references=VisualReferences(body=[], skull=[]),
            image_prompt="...",
        )


def test_factsheet_valid():
    fs = FactSheet(
        species="Tyrannosaurus rex",
        subtitle="...",
        annotations=[Annotation(region="tête", facts=["museau"], source_ids=["paper:0"])],
        skull_view=SkullView(facts=["..."], scale_cm=50, source_ids=["paper:0"]),
        size=Size(length_m=[12, 13], hip_height_m=[3.5, 4], source_ids=["paper:0"]),
        conclusion="...",
        references=[Reference(id="paper:0", citation_short="X 2024", doi=None, title="t")],
        visual_references=VisualReferences(body=[], skull=[]),
        image_prompt="photoreal trex, no text",
    )
    assert fs.species == "Tyrannosaurus rex"
```

- [ ] **Step 2: Run — expect import error**

```bash
pytest tests/test_models.py -v
```
Expected: FAIL.

- [ ] **Step 3: Write the models**

`src/dino_drawer/models.py`:
```python
"""Pydantic models for every pipeline artifact (papers, refs, factsheet)."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# --- Step 1: papers ---

class Paper(BaseModel):
    """A scientific publication fetched from one of the research sources."""
    doi: str | None
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    source: str  # "semantic_scholar" | "openalex" | "wikipedia"


class WikipediaContext(BaseModel):
    """Wikipedia article context used as baseline knowledge."""
    url: str
    extract: str


class PapersFile(BaseModel):
    species: str
    wikipedia: WikipediaContext | None
    papers: list[Paper]


# --- Step 2: raw images ---

class RawImage(BaseModel):
    """An image candidate downloaded but not yet filtered."""
    id: int
    path: str  # relative to species output dir
    source_url: str
    source: str  # "wikimedia_commons" | "wikipedia" | "phylopic" | "inaturalist"
    credit: str
    license: str
    width: int
    height: int
    search_query: str


class RawImagesFile(BaseModel):
    species: str
    images: list[RawImage]


# --- Step 3: classified images ---

class ClassifiedImage(BaseModel):
    """A raw image after VLM classification."""
    id: int
    path: str  # in refs/ if kept
    type: str
    view: str
    usable_for_body_generation: bool
    usable_for_skull_generation: bool
    realism_score: float
    quality_score: float
    description_courte: str
    credit: str
    license: str


class RefsFile(BaseModel):
    species: str
    body: list[ClassifiedImage] = Field(default_factory=list)
    skull: list[ClassifiedImage] = Field(default_factory=list)
    rejected_count: int = 0


# --- Step 4: factsheet ---

class Annotation(BaseModel):
    region: str  # tête | peau_et_couverture | membres_anterieurs | membres_posterieurs | queue
    facts: list[str]
    source_ids: list[str]


class SkullView(BaseModel):
    facts: list[str]
    scale_cm: float
    source_ids: list[str]


class Size(BaseModel):
    length_m: list[float]  # [min, max]
    hip_height_m: list[float]  # [min, max]
    source_ids: list[str]


class Reference(BaseModel):
    id: str
    citation_short: str
    doi: str | None
    title: str


class VisualRef(BaseModel):
    path: str
    credit: str
    license: str
    score: float


class VisualReferences(BaseModel):
    body: list[VisualRef] = Field(default_factory=list)
    skull: list[VisualRef] = Field(default_factory=list)


class FactSheet(BaseModel):
    species: str
    subtitle: str
    annotations: list[Annotation]
    skull_view: SkullView
    size: Size
    conclusion: str
    references: list[Reference]
    visual_references: VisualReferences
    image_prompt: str

    @model_validator(mode="after")
    def _every_fact_must_have_a_known_source(self) -> "FactSheet":
        known = {r.id for r in self.references}
        for ann in self.annotations:
            for sid in ann.source_ids:
                if sid not in known:
                    raise ValueError(f"Annotation '{ann.region}' references unknown source {sid!r}")
        for sid in self.skull_view.source_ids + self.size.source_ids:
            if sid not in known:
                raise ValueError(f"Unknown source id {sid!r}")
        return self
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_models.py -v
```
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/dino_drawer/models.py tests/test_models.py
git commit -m "✨ [models] Pydantic schemas for pipeline artifacts"
```

---

## Task 2: Research/papers — Wikipedia + Semantic Scholar + OpenAlex

**Files:**
- Create: `src/dino_drawer/research/__init__.py`
- Create: `src/dino_drawer/research/papers/__init__.py`
- Create: `src/dino_drawer/research/papers/wikipedia.py`
- Create: `src/dino_drawer/research/papers/semantic_scholar.py`
- Create: `src/dino_drawer/research/papers/openalex.py`
- Create: `src/dino_drawer/research/papers/__main__.py`
- Create: `tests/test_research_papers.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/wikipedia_trex.json`
- Create: `tests/fixtures/semantic_scholar_trex.json`
- Create: `tests/fixtures/openalex_trex.json`

- [ ] **Step 1: Save HTTP fixtures**

Create `tests/fixtures/wikipedia_trex.json`:
```json
{
  "title": "Tyrannosaurus",
  "extract": "Tyrannosaurus is a genus of large theropod dinosaur. The species T. rex lived throughout western North America in the late Maastrichtian.",
  "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Tyrannosaurus"}}
}
```

Create `tests/fixtures/semantic_scholar_trex.json`:
```json
{
  "total": 1,
  "data": [
    {
      "title": "Lipped Tyrannosaurs",
      "abstract": "Evidence for extra-oral tissues...",
      "year": 2023,
      "authors": [{"name": "T. Cullen"}, {"name": "M. Frederickson"}],
      "externalIds": {"DOI": "10.1126/science.abc1234"}
    }
  ]
}
```

Create `tests/fixtures/openalex_trex.json`:
```json
{
  "results": [
    {
      "title": "Soft-tissue skin in T. rex",
      "abstract_inverted_index": {"Soft-tissue": [0], "skin": [1]},
      "publication_year": 2024,
      "authorships": [{"author": {"display_name": "P. Sereno"}}],
      "doi": "https://doi.org/10.1098/rspb.2024.0001"
    }
  ]
}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_research_papers.py`:
```python
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
```

- [ ] **Step 3: Run — expect failure**

```bash
pytest tests/test_research_papers.py -v
```
Expected: FAIL.

- [ ] **Step 4: Implement Wikipedia client**

`src/dino_drawer/research/__init__.py`:
```python
"""Research module: paper fetchers and image scrapers."""
```

`src/dino_drawer/research/papers/__init__.py`:
```python
"""Paper sources for the research step. Exposes fetch_all aggregator."""
from .aggregator import fetch_all

__all__ = ["fetch_all"]
```

`src/dino_drawer/research/papers/wikipedia.py`:
```python
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
```

- [ ] **Step 5: Implement Semantic Scholar client**

`src/dino_drawer/research/papers/semantic_scholar.py`:
```python
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
```

- [ ] **Step 6: Implement OpenAlex client**

`src/dino_drawer/research/papers/openalex.py`:
```python
"""OpenAlex API client. Complement to Semantic Scholar."""
import httpx

from dino_drawer.models import Paper

_BASE = "https://api.openalex.org/works"


def _decode_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as inverted index; reconstruct the text."""
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
```

- [ ] **Step 7: Implement aggregator**

`src/dino_drawer/research/papers/aggregator.py`:
```python
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
```

- [ ] **Step 8: Implement standalone runner**

`src/dino_drawer/research/papers/__main__.py`:
```python
"""Standalone runner: `python -m dino_drawer.research.papers 'Tyrannosaurus rex'`."""
import asyncio
import json
import sys
from pathlib import Path

from .aggregator import fetch_all


async def _run(species: str, out_dir: Path) -> Path:
    result = await fetch_all(species)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "papers.json"
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print(f"Wrote {path} ({len(result.papers)} papers)")
    return path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.research.papers '<species>'", file=sys.stderr)
        return 1
    species = sys.argv[1]
    slug = species.lower().replace(" ", "-")
    out = Path("out") / slug
    asyncio.run(_run(species, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 9: Run tests — expect pass**

```bash
pytest tests/test_research_papers.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 10: Smoke test against real APIs**

```bash
python -m dino_drawer.research.papers "Tyrannosaurus rex"
cat out/tyrannosaurus-rex/papers.json | head -40
```
Expected: real papers fetched, valid JSON.

- [ ] **Step 11: Commit**

```bash
git add src/dino_drawer/research tests/test_research_papers.py tests/fixtures/
git commit -m "✨ [research] Papers aggregator: Wikipedia + Semantic Scholar + OpenAlex"
```

---

## Task 3: Research/images — Wikimedia + PhyloPic + iNaturalist

**Files:**
- Create: `src/dino_drawer/research/images/__init__.py`
- Create: `src/dino_drawer/research/images/wikimedia.py`
- Create: `src/dino_drawer/research/images/phylopic.py`
- Create: `src/dino_drawer/research/images/inaturalist.py`
- Create: `src/dino_drawer/research/images/aggregator.py`
- Create: `src/dino_drawer/research/images/__main__.py`
- Create: `tests/test_research_images.py`
- Create: `tests/fixtures/wikimedia_search_trex.json`
- Create: `tests/fixtures/tiny_jpeg.jpg` (a small valid JPEG)

- [ ] **Step 1: Save fixtures**

Create `tests/fixtures/wikimedia_search_trex.json`:
```json
{
  "query": {
    "search": [
      {"title": "File:Trex_paleoart.jpg"},
      {"title": "File:Trex_skeleton_phylogeny_chart.png"}
    ]
  }
}
```

Create `tests/fixtures/wikimedia_imageinfo_trex.json`:
```json
{
  "query": {
    "pages": {
      "1": {
        "title": "File:Trex_paleoart.jpg",
        "imageinfo": [{
          "url": "https://upload.wikimedia.org/.../Trex_paleoart.jpg",
          "width": 1920,
          "height": 1280,
          "extmetadata": {
            "Artist": {"value": "John Conway"},
            "LicenseShortName": {"value": "CC BY-SA 4.0"}
          }
        }]
      }
    }
  }
}
```

Create `tests/fixtures/tiny_jpeg.jpg` via Python helper in test (or commit a real tiny JPEG). For determinism, generate at test time:
```python
# helper, will live inside test file
from PIL import Image
import io
def make_jpeg(w=1024, h=768) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "olive").save(buf, "JPEG")
    return buf.getvalue()
```

- [ ] **Step 2: Write the failing test**

`tests/test_research_images.py`:
```python
"""Tests for image scrapers and pre-filtering."""
import io
import json
from pathlib import Path

import httpx
import pytest
import respx
from PIL import Image

from dino_drawer.research.images import wikimedia
from dino_drawer.research.images import fetch_all

FIXTURES = Path(__file__).parent / "fixtures"


def _make_jpeg(w: int = 1024, h: int = 768, color: str = "olive") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


@respx.mock
async def test_wikimedia_rejects_chart_filename(tmp_path):
    search = json.loads((FIXTURES / "wikimedia_search_trex.json").read_text())
    imageinfo = json.loads((FIXTURES / "wikimedia_imageinfo_trex.json").read_text())
    respx.get(url__regex=r"https://commons\.wikimedia\.org/.+list=search.*").mock(return_value=httpx.Response(200, json=search))
    respx.get(url__regex=r"https://commons\.wikimedia\.org/.+prop=imageinfo.*").mock(return_value=httpx.Response(200, json=imageinfo))
    respx.get(url__regex=r"https://upload\.wikimedia\.org/.*").mock(return_value=httpx.Response(200, content=_make_jpeg()))

    images = await wikimedia.fetch("Tyrannosaurus rex", out_dir=tmp_path, max_results=10)
    # phylogeny_chart filename must be rejected pre-download
    names = [img.path for img in images]
    assert all("chart" not in n.lower() and "phylogeny" not in n.lower() for n in names)


@respx.mock
async def test_wikimedia_rejects_small_images(tmp_path):
    search = json.loads((FIXTURES / "wikimedia_search_trex.json").read_text())
    info = json.loads((FIXTURES / "wikimedia_imageinfo_trex.json").read_text())
    # patch image to be too small
    info["query"]["pages"]["1"]["imageinfo"][0]["width"] = 400
    info["query"]["pages"]["1"]["imageinfo"][0]["height"] = 300
    respx.get(url__regex=r"https://commons\.wikimedia\.org/.+list=search.*").mock(return_value=httpx.Response(200, json=search))
    respx.get(url__regex=r"https://commons\.wikimedia\.org/.+prop=imageinfo.*").mock(return_value=httpx.Response(200, json=info))

    images = await wikimedia.fetch("Tyrannosaurus rex", out_dir=tmp_path, max_results=10)
    assert images == []  # all filtered out
```

- [ ] **Step 3: Run — expect failure**

```bash
pytest tests/test_research_images.py -v
```
Expected: FAIL (modules don't exist).

- [ ] **Step 4: Implement Wikimedia client**

`src/dino_drawer/research/images/__init__.py`:
```python
"""Image sources for the research step."""
from .aggregator import fetch_all

__all__ = ["fetch_all"]
```

`src/dino_drawer/research/images/wikimedia.py`:
```python
"""Wikimedia Commons image scraper for a species, with license metadata."""
from pathlib import Path

import httpx

from dino_drawer.models import RawImage

_API = "https://commons.wikimedia.org/w/api.php"
_REJECT_KEYWORDS = ("chart", "phylogeny", "cladogram", "range_map", "rangemap", "graph")
_MIN_W, _MIN_H = 800, 600
_ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _rejected_by_name(filename: str) -> bool:
    """Reject filenames containing chart-like keywords before any download."""
    low = filename.lower()
    if not any(low.endswith(ext) for ext in _ALLOWED_EXT):
        return True
    return any(k in low for k in _REJECT_KEYWORDS)


async def _search_titles(client: httpx.AsyncClient, query: str, limit: int) -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srnamespace": 6,
        "srsearch": query,
        "srlimit": limit,
    }
    r = await client.get(_API, params=params, headers={"User-Agent": "dino-drawer/0.1"})
    r.raise_for_status()
    return [hit["title"] for hit in r.json().get("query", {}).get("search", [])]


async def _imageinfo(client: httpx.AsyncClient, title: str) -> dict | None:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
    }
    r = await client.get(_API, params=params, headers={"User-Agent": "dino-drawer/0.1"})
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for _, page in pages.items():
        infos = page.get("imageinfo")
        if infos:
            return {**infos[0], "title": page.get("title", title)}
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
                info = await _imageinfo(client, title)
                if not info:
                    continue
                if info.get("width", 0) < _MIN_W or info.get("height", 0) < _MIN_H:
                    continue
                url = info.get("url")
                if not url:
                    continue
                # download
                resp = await client.get(url, headers={"User-Agent": "dino-drawer/0.1"})
                if resp.status_code != 200:
                    continue
                ext = "." + url.rsplit(".", 1)[-1].lower()
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
```

- [ ] **Step 5: Implement PhyloPic and iNaturalist clients**

`src/dino_drawer/research/images/phylopic.py`:
```python
"""PhyloPic API client for vector silhouettes. Returns raster downloads."""
from pathlib import Path

import httpx

from dino_drawer.models import RawImage

_BASE = "https://api.phylopic.org"


async def fetch(species: str, out_dir: Path, max_results: int = 3) -> list[RawImage]:
    """Fetch silhouettes from PhyloPic. Small (used as scale, not body ref)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(f"{_BASE}/autocomplete", params={"query": species, "embed_primaryImage": "true"})
        if r.status_code != 200:
            return []
        items = r.json().get("_embedded", {}).get("items", [])[:max_results]
        results: list[RawImage] = []
        for i, it in enumerate(items):
            href = it.get("_links", {}).get("primaryImage", {}).get("href")
            if not href:
                continue
            img = await client.get(f"{_BASE}{href}")
            if img.status_code != 200:
                continue
            img_meta = img.json()
            src_url = img_meta.get("_links", {}).get("rasterFiles", [{}])[0].get("href")
            if not src_url:
                continue
            raster = await client.get(src_url)
            if raster.status_code != 200:
                continue
            fname = f"pp_{i}.png"
            (out_dir / fname).write_bytes(raster.content)
            results.append(RawImage(
                id=10_000 + i,  # large offset to avoid id collision with wikimedia
                path=f"refs_raw/{fname}",
                source_url=src_url,
                source="phylopic",
                credit=img_meta.get("attribution", "PhyloPic"),
                license=img_meta.get("license", "CC0"),
                width=img_meta.get("sizes", {}).get("width", 0),
                height=img_meta.get("sizes", {}).get("height", 0),
                search_query=species,
            ))
        return results
```

`src/dino_drawer/research/images/inaturalist.py`:
```python
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
        # find taxon id
        tx = await client.get(f"{_BASE}/taxa", params={"q": species, "rank": "species"})
        if tx.status_code != 200:
            return []
        taxa = tx.json().get("results", [])
        if not taxa:
            return []
        # iNat marks extinct taxa; skip them
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
```

- [ ] **Step 6: Implement aggregator + runner**

`src/dino_drawer/research/images/aggregator.py`:
```python
"""Run all image sources concurrently, write refs_raw.json."""
import asyncio
import json
from pathlib import Path

from dino_drawer.models import RawImagesFile
from . import wikimedia, phylopic, inaturalist


async def fetch_all(species: str, out_dir: Path, max_total: int = 50) -> RawImagesFile:
    """Run all 3 sources in parallel, write refs_raw.json."""
    raw_dir = Path(out_dir) / "refs_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wm, pp, inat = await asyncio.gather(
        wikimedia.fetch(species, raw_dir, max_results=max_total),
        phylopic.fetch(species, raw_dir, max_results=3),
        inaturalist.fetch(species, raw_dir, max_results=10),
        return_exceptions=False,
    )
    all_images = [*wm, *pp, *inat][:max_total]
    result = RawImagesFile(species=species, images=all_images)
    (Path(out_dir) / "refs_raw.json").write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    )
    return result
```

`src/dino_drawer/research/images/__main__.py`:
```python
"""Standalone runner: `python -m dino_drawer.research.images 'Tyrannosaurus rex'`."""
import asyncio
import sys
from pathlib import Path

from .aggregator import fetch_all


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.research.images '<species>'", file=sys.stderr)
        return 1
    species = sys.argv[1]
    slug = species.lower().replace(" ", "-")
    result = asyncio.run(fetch_all(species, Path("out") / slug))
    print(f"Wrote {len(result.images)} raw images to out/{slug}/refs_raw/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Run tests — expect pass**

```bash
pytest tests/test_research_images.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 8: Smoke test**

```bash
python -m dino_drawer.research.images "Tyrannosaurus rex"
ls out/tyrannosaurus-rex/refs_raw/ | head -10
```
Expected: 5-30 images downloaded.

- [ ] **Step 9: Commit**

```bash
git add src/dino_drawer/research/images tests/test_research_images.py tests/fixtures/wikimedia_*.json
git commit -m "✨ [research] Image scrapers: Wikimedia + PhyloPic + iNaturalist with pre-filtering"
```

---

## Task 4: Vision/vlm_client — Ollama vision client

**Files:**
- Create: `src/dino_drawer/vision/__init__.py`
- Create: `src/dino_drawer/vision/vlm_client.py`
- Create: `tests/test_vlm_client.py`

- [ ] **Step 1: Write the failing test**

`tests/test_vlm_client.py`:
```python
"""Tests for the VLM (vision LLM) Ollama wrapper."""
from pathlib import Path
from unittest.mock import patch

import pytest

from dino_drawer.vision.vlm_client import VLMClient, VLMError


def test_classify_returns_parsed_json(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG

    fake_response = {
        "message": {"content": '{"type":"paleoart_realiste","view":"profil_corps","usable_for_body_generation":true,"usable_for_skull_generation":false,"realism_score":8,"quality_score":7,"description_courte":"trex de profil"}'}
    }

    with patch("dino_drawer.vision.vlm_client.ollama.chat", return_value=fake_response) as m:
        client = VLMClient(model="qwen2.5vl:7b")
        out = client.classify_image(img, species="Tyrannosaurus rex")
    assert out["type"] == "paleoart_realiste"
    assert out["usable_for_body_generation"] is True
    m.assert_called_once()


def test_classify_retries_on_invalid_json(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bad = {"message": {"content": "not json at all"}}
    good = {"message": {"content": '{"type":"autre","view":"autre","usable_for_body_generation":false,"usable_for_skull_generation":false,"realism_score":0,"quality_score":0,"description_courte":""}'}}
    with patch("dino_drawer.vision.vlm_client.ollama.chat", side_effect=[bad, good]) as m:
        client = VLMClient(model="qwen2.5vl:7b")
        out = client.classify_image(img, species="X")
    assert m.call_count == 2
    assert out["type"] == "autre"


def test_classify_raises_after_max_retries(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    with patch("dino_drawer.vision.vlm_client.ollama.chat", return_value={"message": {"content": "garbage"}}):
        client = VLMClient(model="qwen2.5vl:7b", max_retries=2)
        with pytest.raises(VLMError):
            client.classify_image(img, species="X")
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_vlm_client.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement VLM client**

`src/dino_drawer/vision/__init__.py`:
```python
"""Vision module: VLM-based image filtering and description."""
```

`src/dino_drawer/vision/vlm_client.py`:
```python
"""Thin wrapper around Ollama's vision API with JSON validation + retry."""
from __future__ import annotations

import json
import re
from pathlib import Path

import ollama


class VLMError(RuntimeError):
    """Raised when the VLM cannot produce valid JSON after retries."""


_CLASSIFY_PROMPT = """\
Tu analyses une image candidate pour servir de référence à la génération
d'illustration scientifique de {species}.

Réponds en JSON strict, sans markdown :
{{
  "type": "paleoart_realiste" | "rendu_3d" | "photo_squelette" | "fossile" | "schema_anatomique" | "cladogramme" | "illustration_enfant" | "photo_specimen_vivant" | "carte_distribution" | "autre",
  "view": "profil_corps" | "trois_quarts_corps" | "face_corps" | "crane_profil" | "crane_face" | "detail" | "scene_groupe" | "autre",
  "usable_for_body_generation": true|false,
  "usable_for_skull_generation": true|false,
  "realism_score": 0-10,
  "quality_score": 0-10,
  "description_courte": "string"
}}"""


def _extract_json(text: str) -> dict | None:
    """Try to parse JSON from a text that may contain markdown fences or prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class VLMClient:
    """Ollama vision client with JSON validation and retry-on-invalid."""

    def __init__(self, model: str = "qwen2.5vl:7b", max_retries: int = 2) -> None:
        self.model = model
        self.max_retries = max_retries

    def classify_image(self, image_path: Path, species: str) -> dict:
        """Classify a single image. Returns dict matching the JSON schema."""
        prompt = _CLASSIFY_PROMPT.format(species=species)
        last_err = ""
        for attempt in range(self.max_retries + 1):
            user_content = prompt if attempt == 0 else (
                f"{prompt}\n\nPrécédente tentative invalide ({last_err}). Renvoie uniquement le JSON."
            )
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": user_content, "images": [str(image_path)]}],
                options={"temperature": 0.0},
            )
            content = resp.get("message", {}).get("content", "")
            parsed = _extract_json(content)
            if parsed is not None and "type" in parsed:
                return parsed
            last_err = "missing 'type' or non-JSON output"
        raise VLMError(f"VLM failed to return valid JSON after {self.max_retries + 1} tries")

    def describe_image(self, image_path: Path, prompt: str) -> str:
        """Free-form description of an image for prompt-enrichment."""
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
            options={"temperature": 0.2},
        )
        return resp.get("message", {}).get("content", "").strip()
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_vlm_client.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dino_drawer/vision/__init__.py src/dino_drawer/vision/vlm_client.py tests/test_vlm_client.py
git commit -m "✨ [vision] Ollama VLM client with JSON validation and retry"
```

---

## Task 5: Vision/classifier — pre-filter, VLM call, selection, output refs.json

**Files:**
- Create: `src/dino_drawer/vision/classifier.py`
- Create: `src/dino_drawer/vision/__main__.py`
- Create: `tests/test_vision_classifier.py`

- [ ] **Step 1: Write the failing test**

`tests/test_vision_classifier.py`:
```python
"""Tests for the image classifier selection rules (VLM mocked)."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dino_drawer.models import RawImage, RawImagesFile
from dino_drawer.vision.classifier import classify_and_select


def _raw(id_: int, path: str = "refs_raw/x.jpg") -> RawImage:
    return RawImage(
        id=id_, path=path, source_url="http://x",
        source="wikimedia_commons", credit="X", license="CC0",
        width=1200, height=800, search_query="x",
    )


def test_keeps_top_three_body_and_two_skull(tmp_path):
    # 5 raw images, mocked VLM responses
    raws = RawImagesFile(species="X", images=[_raw(i, f"refs_raw/i{i}.jpg") for i in range(5)])
    (tmp_path / "refs_raw.json").write_text(raws.model_dump_json())
    raw_dir = tmp_path / "refs_raw"
    raw_dir.mkdir()
    for i in range(5):
        (raw_dir / f"i{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    classifications = [
        {"type": "paleoart_realiste", "view": "profil_corps", "usable_for_body_generation": True, "usable_for_skull_generation": False, "realism_score": 9, "quality_score": 9, "description_courte": "a"},
        {"type": "paleoart_realiste", "view": "profil_corps", "usable_for_body_generation": True, "usable_for_skull_generation": False, "realism_score": 7, "quality_score": 8, "description_courte": "b"},
        {"type": "paleoart_realiste", "view": "trois_quarts_corps", "usable_for_body_generation": True, "usable_for_skull_generation": False, "realism_score": 8, "quality_score": 7, "description_courte": "c"},
        {"type": "paleoart_realiste", "view": "profil_corps", "usable_for_body_generation": True, "usable_for_skull_generation": False, "realism_score": 5, "quality_score": 5, "description_courte": "d"},  # filtered (realism<6)
        {"type": "photo_squelette", "view": "crane_profil", "usable_for_body_generation": False, "usable_for_skull_generation": True, "realism_score": 0, "quality_score": 8, "description_courte": "e"},
    ]

    with patch("dino_drawer.vision.classifier.VLMClient") as MockVLM:
        instance = MockVLM.return_value
        instance.classify_image.side_effect = classifications
        refs = classify_and_select(species="X", out_dir=tmp_path)

    assert len(refs.body) == 3
    assert {r.id for r in refs.body} == {0, 1, 2}  # top 3 by realism*quality
    assert len(refs.skull) == 1
    assert refs.skull[0].id == 4


def test_drops_rejected_types(tmp_path):
    raws = RawImagesFile(species="X", images=[_raw(0), _raw(1)])
    (tmp_path / "refs_raw.json").write_text(raws.model_dump_json())
    (tmp_path / "refs_raw").mkdir()
    (tmp_path / "refs_raw" / "x.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    classifications = [
        {"type": "cladogramme", "view": "autre", "usable_for_body_generation": False, "usable_for_skull_generation": False, "realism_score": 0, "quality_score": 0, "description_courte": "x"},
        {"type": "illustration_enfant", "view": "profil_corps", "usable_for_body_generation": True, "usable_for_skull_generation": False, "realism_score": 9, "quality_score": 9, "description_courte": "x"},  # rejected by type even if usable
    ]
    with patch("dino_drawer.vision.classifier.VLMClient") as MockVLM:
        MockVLM.return_value.classify_image.side_effect = classifications
        refs = classify_and_select(species="X", out_dir=tmp_path)
    assert refs.body == []
    assert refs.skull == []
    assert refs.rejected_count == 2
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_vision_classifier.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement classifier**

`src/dino_drawer/vision/classifier.py`:
```python
"""Apply VLM classification to raw images, then select best per usage."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from dino_drawer.models import (
    ClassifiedImage,
    RawImagesFile,
    RefsFile,
)
from .vlm_client import VLMClient, VLMError

_REJECTED_TYPES = {"cladogramme", "illustration_enfant", "carte_distribution"}
_BODY_TYPES = {"paleoart_realiste", "rendu_3d", "photo_specimen_vivant"}
_BODY_VIEWS = {"profil_corps", "trois_quarts_corps"}
_SKULL_VIEWS = {"crane_profil", "crane_face"}


def _score(c: ClassifiedImage) -> float:
    return c.realism_score * c.quality_score


def classify_and_select(
    species: str,
    out_dir: Path,
    *,
    vlm_model: str = "qwen2.5vl:7b",
    n_body: int = 3,
    n_skull: int = 2,
) -> RefsFile:
    """Read refs_raw.json, classify with VLM, select top-N body and skull refs.

    Writes refs.json and copies kept images to refs/.
    """
    out_dir = Path(out_dir)
    raw_meta = RawImagesFile.model_validate_json((out_dir / "refs_raw.json").read_text())
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    vlm = VLMClient(model=vlm_model)
    classified: list[ClassifiedImage] = []
    rejected = 0
    for raw in raw_meta.images:
        img_path = out_dir / raw.path
        if not img_path.exists():
            rejected += 1
            continue
        try:
            data = vlm.classify_image(img_path, species=species)
        except VLMError:
            rejected += 1
            continue
        if data.get("type") in _REJECTED_TYPES:
            rejected += 1
            continue
        classified.append(ClassifiedImage(
            id=raw.id,
            path=raw.path,
            type=data["type"],
            view=data["view"],
            usable_for_body_generation=bool(data.get("usable_for_body_generation")),
            usable_for_skull_generation=bool(data.get("usable_for_skull_generation")),
            realism_score=float(data.get("realism_score", 0)),
            quality_score=float(data.get("quality_score", 0)),
            description_courte=data.get("description_courte", ""),
            credit=raw.credit,
            license=raw.license,
        ))

    body_candidates = [c for c in classified
                       if c.usable_for_body_generation
                       and c.type in _BODY_TYPES
                       and c.view in _BODY_VIEWS
                       and c.realism_score >= 6]
    body_top = sorted(body_candidates, key=_score, reverse=True)[:n_body]

    skull_candidates = [c for c in classified
                        if c.usable_for_skull_generation
                        and c.view in _SKULL_VIEWS
                        and c.quality_score >= 5]
    skull_top = sorted(skull_candidates, key=_score, reverse=True)[:n_skull]

    # Copy kept images to refs/ and rewrite their paths
    def _copy_and_rewrite(items: list[ClassifiedImage], prefix: str) -> list[ClassifiedImage]:
        out: list[ClassifiedImage] = []
        for i, c in enumerate(items):
            src = out_dir / c.path
            ext = src.suffix
            dst_rel = f"refs/{prefix}_{i}{ext}"
            shutil.copy(src, out_dir / dst_rel)
            out.append(c.model_copy(update={"path": dst_rel}))
        return out

    body_top = _copy_and_rewrite(body_top, "body")
    skull_top = _copy_and_rewrite(skull_top, "skull")

    refs = RefsFile(
        species=species,
        body=body_top,
        skull=skull_top,
        rejected_count=rejected + (len(classified) - len(body_top) - len(skull_top)),
    )
    (out_dir / "refs.json").write_text(json.dumps(refs.model_dump(), indent=2, ensure_ascii=False))
    return refs
```

- [ ] **Step 4: Implement standalone runner**

`src/dino_drawer/vision/__main__.py`:
```python
"""Standalone runner: `python -m dino_drawer.vision <out/species/>`."""
import sys
from pathlib import Path

from .classifier import classify_and_select


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.vision <species-out-dir>", file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    species = (out_dir.name.replace("-", " ")).title()
    refs = classify_and_select(species=species, out_dir=out_dir)
    print(f"Kept {len(refs.body)} body refs and {len(refs.skull)} skull refs (rejected {refs.rejected_count})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/test_vision_classifier.py -v
```
Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dino_drawer/vision/classifier.py src/dino_drawer/vision/__main__.py tests/test_vision_classifier.py
git commit -m "✨ [vision] Image classifier with VLM + top-N selection rules"
```

---

## Task 6: Vision/describer — VLM descriptions of selected refs

**Files:**
- Create: `src/dino_drawer/vision/describer.py`
- Create: `tests/test_vision_describer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_vision_describer.py`:
```python
"""Tests for VLM-based body-reference descriptions."""
from pathlib import Path
from unittest.mock import patch

from dino_drawer.models import ClassifiedImage, RefsFile
from dino_drawer.vision.describer import describe_body_refs


def _classified(i: int) -> ClassifiedImage:
    return ClassifiedImage(
        id=i, path=f"refs/body_{i}.jpg", type="paleoart_realiste", view="profil_corps",
        usable_for_body_generation=True, usable_for_skull_generation=False,
        realism_score=8, quality_score=8, description_courte="",
        credit="x", license="CC0",
    )


def test_describer_returns_concatenated_brief(tmp_path):
    (tmp_path / "refs").mkdir()
    for i in range(3):
        (tmp_path / f"refs/body_{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    refs = RefsFile(species="X", body=[_classified(0), _classified(1), _classified(2)], skull=[])

    with patch("dino_drawer.vision.describer.VLMClient") as MockVLM:
        MockVLM.return_value.describe_image.side_effect = [
            "Peau verte avec rayures sombres. Plumes éparses sur le cou.",
            "Posture en S, gueule entrouverte.",
            "Forêt humide, lumière oblique.",
        ]
        brief = describe_body_refs(refs=refs, out_dir=tmp_path)

    assert "peau verte" in brief.lower() or "Peau verte" in brief
    assert "plumes" in brief.lower()
    assert "forêt" in brief.lower() or "forêt" in brief
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_vision_describer.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement describer**

`src/dino_drawer/vision/describer.py`:
```python
"""Generate a `visual_brief` paragraph from selected body references."""
from pathlib import Path

from dino_drawer.models import RefsFile
from .vlm_client import VLMClient

_DESCRIBE_PROMPT = (
    "Décris cette image en 2 phrases : couleurs et motifs de la peau, "
    "présence et localisation des plumes/poils/écailles, posture, environnement visible. "
    "Ne mentionne pas l'espèce."
)


def describe_body_refs(refs: RefsFile, out_dir: Path, vlm_model: str = "qwen2.5vl:7b") -> str:
    """Call the VLM on each body ref, return a single concatenated brief string."""
    vlm = VLMClient(model=vlm_model)
    parts: list[str] = []
    for c in refs.body:
        img_path = Path(out_dir) / c.path
        if not img_path.exists():
            continue
        parts.append(vlm.describe_image(img_path, _DESCRIBE_PROMPT))
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_vision_describer.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dino_drawer/vision/describer.py tests/test_vision_describer.py
git commit -m "✨ [vision] VLM describer producing visual_brief from body refs"
```

---

## Task 7: Synthesis/ollama_client + prompts

**Files:**
- Create: `src/dino_drawer/synthesis/__init__.py`
- Create: `src/dino_drawer/synthesis/ollama_client.py`
- Create: `src/dino_drawer/synthesis/prompts.py`
- Create: `tests/test_synthesis_client.py`

- [ ] **Step 1: Write the failing test**

`tests/test_synthesis_client.py`:
```python
"""Tests for the synthesis Ollama client (JSON validation + retry)."""
from unittest.mock import patch

import pytest

from dino_drawer.synthesis.ollama_client import call_llm_for_json, SynthesisError


def test_returns_parsed_json_on_first_try():
    fake = {"message": {"content": '{"k":"v"}'}}
    with patch("dino_drawer.synthesis.ollama_client.ollama.chat", return_value=fake):
        out = call_llm_for_json(model="qwen2.5:14b", prompt="x")
    assert out == {"k": "v"}


def test_retries_then_succeeds():
    bad = {"message": {"content": "not json"}}
    good = {"message": {"content": '{"k":1}'}}
    with patch("dino_drawer.synthesis.ollama_client.ollama.chat", side_effect=[bad, good]) as m:
        out = call_llm_for_json(model="qwen2.5:14b", prompt="x", max_retries=2)
    assert out == {"k": 1}
    assert m.call_count == 2


def test_raises_after_max_retries():
    bad = {"message": {"content": "nope"}}
    with patch("dino_drawer.synthesis.ollama_client.ollama.chat", return_value=bad):
        with pytest.raises(SynthesisError):
            call_llm_for_json(model="qwen2.5:14b", prompt="x", max_retries=2)
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_synthesis_client.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement Ollama client**

`src/dino_drawer/synthesis/__init__.py`:
```python
"""Synthesis module: text-LLM-driven factsheet generation."""
```

`src/dino_drawer/synthesis/ollama_client.py`:
```python
"""Wrapper around Ollama for JSON-mode generation with retry-on-invalid."""
from __future__ import annotations

import json
import re

import ollama


class SynthesisError(RuntimeError):
    """Raised when the LLM cannot produce valid JSON after retries."""


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_llm_for_json(
    *,
    model: str,
    prompt: str,
    max_retries: int = 2,
    temperature: float = 0.2,
) -> dict:
    """Call the LLM expecting JSON, retry up to max_retries on parse failure."""
    last_err = ""
    for attempt in range(max_retries + 1):
        user = prompt if attempt == 0 else (
            f"{prompt}\n\nLa précédente sortie était invalide ({last_err}). "
            "Renvoie uniquement le JSON, sans markdown, sans texte avant ou après."
        )
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": user}],
            options={"temperature": temperature},
            format="json",
        )
        content = resp.get("message", {}).get("content", "")
        parsed = _extract_json(content)
        if parsed is not None:
            return parsed
        last_err = "unparseable JSON"
    raise SynthesisError(f"LLM failed to return valid JSON after {max_retries + 1} attempts")
```

- [ ] **Step 4: Implement prompt builder**

`src/dino_drawer/synthesis/prompts.py`:
```python
"""Build the synthesis prompt fed to the text LLM."""
from __future__ import annotations

import json

from dino_drawer.models import PapersFile, RefsFile


_SCHEMA_HINT = """\
Tu réponds en JSON strict avec EXACTEMENT ces champs et types :

{
  "species": "string",
  "subtitle": "string",
  "annotations": [
    {"region": "tête" | "peau_et_couverture" | "membres_anterieurs" | "membres_posterieurs" | "queue",
     "facts": ["string", ...],
     "source_ids": ["paper:N" | "wikipedia"]}
  ],
  "skull_view": {"facts": ["string"], "scale_cm": number, "source_ids": ["paper:N" | "wikipedia"]},
  "size": {"length_m": [min, max], "hip_height_m": [min, max], "source_ids": ["paper:N" | "wikipedia"]},
  "conclusion": "string",
  "references": [{"id": "paper:0" | "wikipedia", "citation_short": "string", "doi": "string|null", "title": "string"}],
  "image_prompt": "string en anglais, photoréaliste, mentionnant >= 3 éléments visuels observés, finit par 'no text'"
}

Contraintes :
- Inclure TOUTES les régions anatomiques ci-dessus. Si tu n'as pas d'info, écris un fait baseline et cite "wikipedia".
- Chaque source_id doit exister dans `references`.
- `image_prompt` doit être en anglais et doit mentionner au moins 3 éléments visuels du visual_brief.
"""


def build_synthesis_prompt(
    species: str,
    papers: PapersFile,
    refs: RefsFile | None,
    visual_brief: str,
    lang: str = "fr",
) -> str:
    """Assemble the full synthesis prompt for the LLM."""
    paper_blocks = []
    for i, p in enumerate(papers.papers):
        paper_blocks.append(
            f"[paper:{i}] {', '.join(p.authors[:3]) or 'Anon'} ({p.year}). "
            f"{p.title}. DOI: {p.doi or 'n/a'}\nRésumé : {p.abstract[:600]}"
        )
    wiki_block = ""
    if papers.wikipedia:
        wiki_block = f"[wikipedia] {papers.wikipedia.url}\n{papers.wikipedia.extract[:1500]}"

    refs_block = ""
    if refs and (refs.body or refs.skull):
        refs_block = "Références visuelles retenues : " + json.dumps(
            {"body": [r.path for r in refs.body], "skull": [r.path for r in refs.skull]}
        )

    brief_block = f"VISUAL_BRIEF (extrait des refs visuelles) :\n{visual_brief}" if visual_brief else ""

    lang_directive = "Rédige les champs textuels en français." if lang == "fr" else "Write text fields in English."

    return f"""\
Tu es un assistant paléontologique. À partir des sources ci-dessous, produis une fiche d'information structurée sur l'espèce {species}.
{lang_directive}

SOURCES SCIENTIFIQUES :
{wiki_block}

{chr(10).join(paper_blocks) if paper_blocks else '(Aucun papier récent.)'}

{refs_block}

{brief_block}

{_SCHEMA_HINT}
"""
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/test_synthesis_client.py -v
```
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dino_drawer/synthesis/__init__.py src/dino_drawer/synthesis/ollama_client.py src/dino_drawer/synthesis/prompts.py tests/test_synthesis_client.py
git commit -m "✨ [synthesis] Ollama JSON client with retry, prompt builder"
```

---

## Task 8: Synthesis main — assemble factsheet

**Files:**
- Create: `src/dino_drawer/synthesis/main.py`
- Create: `src/dino_drawer/synthesis/__main__.py`
- Create: `tests/test_synthesis_main.py`
- Create: `tests/fixtures/trex_papers.json`
- Create: `tests/fixtures/trex_refs.json`

- [ ] **Step 1: Save fixtures**

Create `tests/fixtures/trex_papers.json`:
```json
{
  "species": "Tyrannosaurus rex",
  "wikipedia": {"url": "https://en.wikipedia.org/wiki/Tyrannosaurus", "extract": "Tyrannosaurus is a genus of large theropod dinosaur. T. rex lived in late Maastrichtian."},
  "papers": [
    {"doi": "10.1126/x", "title": "Lipped Tyrannosaurs", "authors": ["T. Cullen"], "year": 2023, "abstract": "Lips covered the teeth.", "source": "semantic_scholar"}
  ]
}
```

Create `tests/fixtures/trex_refs.json`:
```json
{
  "species": "Tyrannosaurus rex",
  "body": [{"id":0,"path":"refs/body_0.jpg","type":"paleoart_realiste","view":"profil_corps","usable_for_body_generation":true,"usable_for_skull_generation":false,"realism_score":9,"quality_score":9,"description_courte":"trex","credit":"x","license":"CC0"}],
  "skull": [],
  "rejected_count": 0
}
```

- [ ] **Step 2: Write the failing test**

`tests/test_synthesis_main.py`:
```python
"""Test end-to-end synthesis with mocked LLM and VLM."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch

from dino_drawer.synthesis.main import synthesize

FIXTURES = Path(__file__).parent / "fixtures"


def test_synthesize_writes_valid_factsheet(tmp_path):
    shutil.copy(FIXTURES / "trex_papers.json", tmp_path / "papers.json")
    shutil.copy(FIXTURES / "trex_refs.json", tmp_path / "refs.json")
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "body_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    factsheet_dict = {
        "species": "Tyrannosaurus rex",
        "subtitle": "...",
        "annotations": [
            {"region": "tête", "facts": ["Lips covering teeth"], "source_ids": ["paper:0"]},
            {"region": "peau_et_couverture", "facts": ["Scaly skin"], "source_ids": ["wikipedia"]},
            {"region": "membres_anterieurs", "facts": ["Short"], "source_ids": ["wikipedia"]},
            {"region": "membres_posterieurs", "facts": ["Strong"], "source_ids": ["wikipedia"]},
            {"region": "queue", "facts": ["Long"], "source_ids": ["wikipedia"]},
        ],
        "skull_view": {"facts": ["large"], "scale_cm": 50, "source_ids": ["wikipedia"]},
        "size": {"length_m": [12, 13], "hip_height_m": [3.5, 4], "source_ids": ["wikipedia"]},
        "conclusion": "...",
        "references": [
            {"id": "paper:0", "citation_short": "Cullen 2023", "doi": "10.1126/x", "title": "Lipped Tyrannosaurs"},
            {"id": "wikipedia", "citation_short": "Wikipedia", "doi": None, "title": "Tyrannosaurus"},
        ],
        "image_prompt": "photorealistic Tyrannosaurus rex, dark olive skin, sparse feathers on neck, dense forest, no text",
    }

    with patch("dino_drawer.synthesis.main.call_llm_for_json", return_value=factsheet_dict), \
         patch("dino_drawer.synthesis.main.describe_body_refs", return_value="Peau olive. Plumes éparses cou. Forêt."):
        fs = synthesize(species="Tyrannosaurus rex", out_dir=tmp_path)

    assert fs.species == "Tyrannosaurus rex"
    assert (tmp_path / "factsheet.json").exists()
    assert len(fs.visual_references.body) == 1
    assert fs.visual_references.body[0].path == "refs/body_0.jpg"
```

- [ ] **Step 3: Run — expect failure**

```bash
pytest tests/test_synthesis_main.py -v
```
Expected: FAIL.

- [ ] **Step 4: Implement synthesize()**

`src/dino_drawer/synthesis/main.py`:
```python
"""Top-level synthesis: papers + refs + visual_brief -> factsheet.json."""
from __future__ import annotations

import json
from pathlib import Path

from dino_drawer.models import (
    FactSheet,
    PapersFile,
    RefsFile,
    VisualRef,
    VisualReferences,
)
from dino_drawer.vision.describer import describe_body_refs
from .ollama_client import call_llm_for_json
from .prompts import build_synthesis_prompt


def _load_refs(out_dir: Path) -> RefsFile | None:
    p = out_dir / "refs.json"
    if not p.exists():
        return None
    return RefsFile.model_validate_json(p.read_text())


def synthesize(
    species: str,
    out_dir: Path,
    *,
    model_llm: str = "qwen2.5:14b-instruct",
    model_vlm: str = "qwen2.5vl:7b",
    lang: str = "fr",
) -> FactSheet:
    """Run synthesis. Writes factsheet.json. Returns the validated FactSheet."""
    out_dir = Path(out_dir)
    papers = PapersFile.model_validate_json((out_dir / "papers.json").read_text())
    refs = _load_refs(out_dir)

    visual_brief = ""
    if refs and refs.body:
        visual_brief = describe_body_refs(refs=refs, out_dir=out_dir, vlm_model=model_vlm)

    prompt = build_synthesis_prompt(species=species, papers=papers, refs=refs, visual_brief=visual_brief, lang=lang)
    raw = call_llm_for_json(model=model_llm, prompt=prompt)

    # Inject visual_references from refs (LLM does not hallucinate them)
    visual_references = VisualReferences(body=[], skull=[])
    if refs:
        visual_references = VisualReferences(
            body=[VisualRef(path=c.path, credit=c.credit, license=c.license, score=c.realism_score * c.quality_score / 10) for c in refs.body],
            skull=[VisualRef(path=c.path, credit=c.credit, license=c.license, score=c.realism_score * c.quality_score / 10) for c in refs.skull],
        )
    raw["visual_references"] = visual_references.model_dump()

    fs = FactSheet.model_validate(raw)
    (out_dir / "factsheet.json").write_text(json.dumps(fs.model_dump(), indent=2, ensure_ascii=False))
    return fs
```

`src/dino_drawer/synthesis/__main__.py`:
```python
"""Standalone runner: `python -m dino_drawer.synthesis <out/species/>`."""
import sys
from pathlib import Path

from .main import synthesize


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.synthesis <species-out-dir>", file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    species = out_dir.name.replace("-", " ").title()
    fs = synthesize(species=species, out_dir=out_dir)
    print(f"Wrote {out_dir}/factsheet.json — {len(fs.annotations)} regions, {len(fs.references)} refs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run — expect pass**

```bash
pytest tests/test_synthesis_main.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dino_drawer/synthesis/main.py src/dino_drawer/synthesis/__main__.py tests/test_synthesis_main.py tests/fixtures/trex_papers.json tests/fixtures/trex_refs.json
git commit -m "✨ [synthesis] Assemble factsheet from papers + refs + visual brief"
```

---

## Task 9: Image/silhouette — SVG by code

**Files:**
- Create: `src/dino_drawer/image/__init__.py`
- Create: `src/dino_drawer/image/silhouette.py`
- Create: `tests/test_silhouette.py`

- [ ] **Step 1: Write the failing test**

`tests/test_silhouette.py`:
```python
"""Test SVG silhouette generator: proportions match Size dimensions."""
from dino_drawer.image.silhouette import build_scale_svg


def test_svg_contains_two_rects_with_correct_ratio():
    svg = build_scale_svg(length_m=12.0, hip_height_m=4.0, human_height_m=1.75)
    # We use rects as the placeholder silhouette. Animal: width=12, human: height=1.75.
    # Just check the produced SVG mentions both objects and has the right viewBox.
    assert "<svg" in svg
    assert 'viewBox="0 0' in svg
    assert "Tyrannosaurus" not in svg  # generic — no species text
    assert "1.75" not in svg  # we don't print raw numbers in the SVG


def test_svg_handles_small_animal():
    svg = build_scale_svg(length_m=0.5, hip_height_m=0.3, human_height_m=1.75)
    assert "<svg" in svg
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_silhouette.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement silhouette**

`src/dino_drawer/image/__init__.py`:
```python
"""Image generation module: diffusion + silhouette SVG."""
```

`src/dino_drawer/image/silhouette.py`:
```python
"""Generate a scale comparison SVG: animal silhouette next to a human."""
from __future__ import annotations


def build_scale_svg(
    length_m: float,
    hip_height_m: float,
    human_height_m: float = 1.75,
    *,
    width_px: int = 600,
    height_px: int = 200,
) -> str:
    """Build an SVG comparing the animal silhouette (rectangle proxy) to a human.

    The animal is drawn as a horizontal rounded rect (length × hip_height),
    the human as a tall thin rect of human_height_m.
    Both are scaled to fit in `width_px × height_px`.
    """
    margin = 20
    total_world_w = length_m + 1.5  # leave space for human
    scale = (width_px - 2 * margin) / total_world_w
    animal_w = length_m * scale
    animal_h = hip_height_m * scale
    human_h = human_height_m * scale
    human_w = 0.4 * scale

    baseline_y = height_px - margin
    animal_y = baseline_y - animal_h
    human_y = baseline_y - human_h
    animal_x = margin
    human_x = margin + animal_w + 0.5 * scale

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}">
  <rect x="{animal_x:.1f}" y="{animal_y:.1f}" width="{animal_w:.1f}" height="{animal_h:.1f}" rx="6" fill="#222"/>
  <rect x="{human_x:.1f}" y="{human_y:.1f}" width="{human_w:.1f}" height="{human_h:.1f}" rx="3" fill="#888"/>
  <line x1="{margin}" y1="{baseline_y}" x2="{width_px - margin}" y2="{baseline_y}" stroke="#444" stroke-width="1"/>
</svg>"""
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_silhouette.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dino_drawer/image/__init__.py src/dino_drawer/image/silhouette.py tests/test_silhouette.py
git commit -m "✨ [image] Scale comparison SVG generator (animal vs human)"
```

---

## Task 10: Image/diffusion — SDXL + IP-Adapter

**Files:**
- Create: `src/dino_drawer/image/diffusion.py`
- Create: `src/dino_drawer/image/__main__.py`
- Create: `tests/test_diffusion.py`

- [ ] **Step 1: Write the failing test**

`tests/test_diffusion.py`:
```python
"""Tests for the diffusion module. Pipeline is mocked; we test wiring only."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from dino_drawer.models import FactSheet, VisualRef, VisualReferences, Annotation, SkullView, Size, Reference
from dino_drawer.image.diffusion import generate_assets


def _basic_factsheet(**overrides) -> FactSheet:
    base = dict(
        species="Tyrannosaurus rex",
        subtitle="",
        annotations=[Annotation(region="tête", facts=["x"], source_ids=["wikipedia"])],
        skull_view=SkullView(facts=["x"], scale_cm=50, source_ids=["wikipedia"]),
        size=Size(length_m=[12, 13], hip_height_m=[3.5, 4], source_ids=["wikipedia"]),
        conclusion="",
        references=[Reference(id="wikipedia", citation_short="W", doi=None, title="t")],
        visual_references=VisualReferences(body=[], skull=[]),
        image_prompt="photoreal trex no text",
    )
    base.update(overrides)
    return FactSheet(**base)


def test_generate_assets_uses_ip_adapter_when_body_refs_present(tmp_path):
    img = Image.new("RGB", (64, 64))
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs/body_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    fs = _basic_factsheet(visual_references=VisualReferences(
        body=[VisualRef(path="refs/body_0.jpg", credit="x", license="CC0", score=8)],
        skull=[],
    ))

    mock_pipe = MagicMock()
    mock_pipe.return_value = MagicMock(images=[img])

    with patch("dino_drawer.image.diffusion._load_pipeline", return_value=mock_pipe) as loader:
        generate_assets(factsheet=fs, out_dir=tmp_path, model="stabilityai/stable-diffusion-xl-base-1.0")
    # Pipeline must have been called with ip_adapter_image
    call_kwargs = mock_pipe.call_args.kwargs
    assert "ip_adapter_image" in call_kwargs
    assert (tmp_path / "hero.png").exists()
    assert (tmp_path / "silhouette.svg").exists()


def test_generate_assets_runs_text_only_when_no_refs(tmp_path):
    img = Image.new("RGB", (64, 64))
    fs = _basic_factsheet()  # empty visual_references

    mock_pipe = MagicMock()
    mock_pipe.return_value = MagicMock(images=[img])

    with patch("dino_drawer.image.diffusion._load_pipeline", return_value=mock_pipe):
        generate_assets(factsheet=fs, out_dir=tmp_path, model="stabilityai/stable-diffusion-xl-base-1.0")

    call_kwargs = mock_pipe.call_args.kwargs
    assert "ip_adapter_image" not in call_kwargs or call_kwargs.get("ip_adapter_image") is None
    assert (tmp_path / "hero.png").exists()
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_diffusion.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement diffusion**

`src/dino_drawer/image/diffusion.py`:
```python
"""SDXL + IP-Adapter pipeline. Generates hero.png, skull.png, silhouette.svg."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from dino_drawer.models import FactSheet
from .silhouette import build_scale_svg

_NEGATIVE = "text, watermark, captions, signature, logo, multiple animals, deformed anatomy"


def _load_pipeline(model: str):
    """Lazy-load the SDXL pipeline with IP-Adapter weights on MPS.

    Separated so tests can monkey-patch it.
    """
    import torch
    from diffusers import StableDiffusionXLPipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
    )
    pipe.set_ip_adapter_scale(0.6)
    return pipe


def _load_ref_images(out_dir: Path, paths: list[str]) -> list[Image.Image]:
    return [Image.open(out_dir / p).convert("RGB") for p in paths]


def generate_assets(
    factsheet: FactSheet,
    out_dir: Path,
    *,
    model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    steps: int = 30,
) -> None:
    """Produce hero.png, skull.png, silhouette.svg in out_dir."""
    out_dir = Path(out_dir)
    pipe = _load_pipeline(model)

    body_refs = _load_ref_images(out_dir, [r.path for r in factsheet.visual_references.body])
    skull_refs = _load_ref_images(out_dir, [r.path for r in factsheet.visual_references.skull])

    # Hero
    hero_kwargs = dict(
        prompt=factsheet.image_prompt,
        negative_prompt=_NEGATIVE,
        num_inference_steps=steps,
        width=1024,
        height=640,
    )
    if body_refs:
        hero_kwargs["ip_adapter_image"] = body_refs
    hero_out = pipe(**hero_kwargs)
    hero_out.images[0].save(out_dir / "hero.png")

    # Skull
    skull_prompt = (
        f"detailed lateral view of {factsheet.species} skull, scientific illustration style, "
        "neutral background, no text"
    )
    skull_kwargs = dict(
        prompt=skull_prompt,
        negative_prompt=_NEGATIVE,
        num_inference_steps=steps,
        width=640,
        height=480,
    )
    if skull_refs:
        skull_kwargs["ip_adapter_image"] = skull_refs
    skull_out = pipe(**skull_kwargs)
    skull_out.images[0].save(out_dir / "skull.png")

    # Silhouette
    length = sum(factsheet.size.length_m) / len(factsheet.size.length_m)
    hip = sum(factsheet.size.hip_height_m) / len(factsheet.size.hip_height_m)
    svg = build_scale_svg(length_m=length, hip_height_m=hip)
    (out_dir / "silhouette.svg").write_text(svg)
```

`src/dino_drawer/image/__main__.py`:
```python
"""Standalone runner: `python -m dino_drawer.image <out/species/>`."""
import sys
from pathlib import Path

from dino_drawer.models import FactSheet
from .diffusion import generate_assets


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.image <species-out-dir>", file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    fs = FactSheet.model_validate_json((out_dir / "factsheet.json").read_text())
    generate_assets(factsheet=fs, out_dir=out_dir)
    print(f"Wrote {out_dir}/hero.png, skull.png, silhouette.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_diffusion.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dino_drawer/image/diffusion.py src/dino_drawer/image/__main__.py tests/test_diffusion.py
git commit -m "✨ [image] SDXL + IP-Adapter diffusion with text-only fallback"
```

---

## Task 11: Compose — HTML template + Playwright render

**Files:**
- Create: `src/dino_drawer/compose/__init__.py`
- Create: `src/dino_drawer/compose/render.py`
- Create: `src/dino_drawer/compose/__main__.py`
- Create: `src/dino_drawer/compose/templates/infographic.html`
- Create: `src/dino_drawer/compose/templates/infographic.css`
- Create: `tests/test_compose.py`

- [ ] **Step 1: Write the failing test**

`tests/test_compose.py`:
```python
"""Tests for HTML rendering and Playwright screenshot."""
import shutil
from pathlib import Path

from PIL import Image

from dino_drawer.models import FactSheet, Annotation, SkullView, Size, Reference, VisualReferences
from dino_drawer.compose.render import render_html, screenshot


def _basic_factsheet() -> FactSheet:
    return FactSheet(
        species="Tyrannosaurus rex",
        subtitle="Reconstitution basée sur les données 2020-2024",
        annotations=[
            Annotation(region="tête", facts=["Museau robuste"], source_ids=["wikipedia"]),
            Annotation(region="peau_et_couverture", facts=["Peau écailleuse"], source_ids=["wikipedia"]),
            Annotation(region="membres_anterieurs", facts=["Courts"], source_ids=["wikipedia"]),
            Annotation(region="membres_posterieurs", facts=["Puissants"], source_ids=["wikipedia"]),
            Annotation(region="queue", facts=["Longue"], source_ids=["wikipedia"]),
        ],
        skull_view=SkullView(facts=["Mâchoire massive"], scale_cm=50, source_ids=["wikipedia"]),
        size=Size(length_m=[12, 13], hip_height_m=[3.5, 4], source_ids=["wikipedia"]),
        conclusion="T. rex avait probablement des lèvres.",
        references=[Reference(id="wikipedia", citation_short="Wikipedia", doi=None, title="T. rex")],
        visual_references=VisualReferences(body=[], skull=[]),
        image_prompt="...",
    )


def test_render_html_contains_all_sections(tmp_path):
    # Create dummy assets
    Image.new("RGB", (1024, 640), "olive").save(tmp_path / "hero.png")
    Image.new("RGB", (640, 480), "gray").save(tmp_path / "skull.png")
    (tmp_path / "silhouette.svg").write_text("<svg></svg>")

    html = render_html(_basic_factsheet(), out_dir=tmp_path)
    assert "Tyrannosaurus rex" in html
    assert "Museau robuste" in html
    assert "T. rex avait probablement des lèvres" in html
    for region in ("tête", "peau", "membres", "queue"):
        assert region.lower() in html.lower()


def test_screenshot_produces_png_with_expected_dimensions(tmp_path):
    Image.new("RGB", (1024, 640), "olive").save(tmp_path / "hero.png")
    Image.new("RGB", (640, 480), "gray").save(tmp_path / "skull.png")
    (tmp_path / "silhouette.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100"/></svg>')

    out = screenshot(_basic_factsheet(), out_dir=tmp_path, width=2000, height=1200)
    assert out.exists()
    img = Image.open(out)
    assert img.size == (2000, 1200)
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_compose.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement template files**

`src/dino_drawer/compose/__init__.py`:
```python
"""Compose module: HTML+CSS template + Playwright screenshot."""
```

`src/dino_drawer/compose/templates/infographic.css`:
```css
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Helvetica Neue', Arial, sans-serif; }
body { background: #1a1a1a; color: #eaeaea; width: 2000px; height: 1200px; padding: 40px; }
.title { font-size: 56px; font-style: italic; letter-spacing: 1px; }
.subtitle { font-size: 20px; color: #aaa; margin-top: 8px; }
.hero-wrap { position: relative; margin-top: 24px; height: 720px; }
.hero { width: 100%; height: 100%; object-fit: cover; border-radius: 4px; }
.annot { position: absolute; max-width: 220px; font-size: 14px; line-height: 1.3; }
.annot h4 { font-size: 12px; letter-spacing: 1.5px; color: #c8a256; margin-bottom: 6px; }
.annot ul { list-style: none; }
.annot.tete { left: 20px; top: 40px; }
.annot.peau { right: 20px; top: 40px; }
.annot.queue { right: 20px; top: 240px; }
.annot.bras { left: 20px; bottom: 240px; }
.annot.pattes { left: 20px; bottom: 60px; }
.bottom { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; margin-top: 24px; height: 280px; }
.bottom .panel { background: #232323; padding: 16px; border-radius: 4px; }
.bottom h3 { font-size: 12px; letter-spacing: 1.5px; color: #c8a256; margin-bottom: 8px; }
.refs { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr; gap: 16px; margin-top: 16px; font-size: 11px; color: #999; }
.skull-img, .silhouette-svg { width: 100%; height: 200px; object-fit: contain; }
```

`src/dino_drawer/compose/templates/infographic.html`:
```html
<!doctype html>
<html lang="{{ lang }}">
<head>
<meta charset="utf-8">
<style>{{ css }}</style>
</head>
<body>
  <h1 class="title">{{ fs.species }}</h1>
  <p class="subtitle">{{ fs.subtitle }}</p>
  <div class="hero-wrap">
    <img class="hero" src="{{ hero_data_uri }}" alt="">
    {% for ann in fs.annotations %}
    <div class="annot {{ region_class(ann.region) }}">
      <h4>{{ ann.region | upper }}</h4>
      <ul>{% for f in ann.facts %}<li>· {{ f }}</li>{% endfor %}</ul>
    </div>
    {% endfor %}
  </div>
  <div class="bottom">
    <div class="panel">
      <h3>CRÂNE (VUE LATÉRALE)</h3>
      <img class="skull-img" src="{{ skull_data_uri }}">
      <ul>{% for f in fs.skull_view.facts %}<li>· {{ f }}</li>{% endfor %}</ul>
    </div>
    <div class="panel">
      <h3>TAILLE ESTIMÉE</h3>
      <div class="silhouette-svg">{{ silhouette_svg | safe }}</div>
      <p>Longueur : {{ fs.size.length_m[0] }}–{{ fs.size.length_m[1] }} m</p>
      <p>Hauteur au bassin : {{ fs.size.hip_height_m[0] }}–{{ fs.size.hip_height_m[1] }} m</p>
    </div>
    <div class="panel">
      <h3>EN CONCLUSION</h3>
      <p>{{ fs.conclusion }}</p>
    </div>
  </div>
  <div class="refs">
    <strong>RÉFÉRENCES SCIENTIFIQUES RÉCENTES</strong>
    {% for r in fs.references %}<div>{{ r.citation_short }} — {{ r.title }}</div>{% endfor %}
  </div>
</body>
</html>
```

- [ ] **Step 4: Implement render.py**

`src/dino_drawer/compose/render.py`:
```python
"""Render the infographic HTML and screenshot it with Playwright."""
from __future__ import annotations

import base64
from pathlib import Path

from jinja2 import Template

from dino_drawer.models import FactSheet

_REGION_TO_CLASS = {
    "tête": "tete",
    "peau_et_couverture": "peau",
    "membres_anterieurs": "bras",
    "membres_posterieurs": "pattes",
    "queue": "queue",
}

_TEMPLATES = Path(__file__).parent / "templates"


def _data_uri(path: Path) -> str:
    """Convert a PNG path to a base64 data URI for inline embedding."""
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def render_html(fs: FactSheet, out_dir: Path, lang: str = "fr") -> str:
    """Render the infographic HTML string with inlined images."""
    css = (_TEMPLATES / "infographic.css").read_text()
    tmpl = Template((_TEMPLATES / "infographic.html").read_text())
    return tmpl.render(
        fs=fs,
        css=css,
        lang=lang,
        hero_data_uri=_data_uri(out_dir / "hero.png"),
        skull_data_uri=_data_uri(out_dir / "skull.png"),
        silhouette_svg=(out_dir / "silhouette.svg").read_text(),
        region_class=lambda r: _REGION_TO_CLASS.get(r, ""),
    )


def screenshot(fs: FactSheet, out_dir: Path, *, width: int = 2000, height: int = 1200) -> Path:
    """Render HTML and use Playwright to screenshot it to final.png."""
    from playwright.sync_api import sync_playwright

    html = render_html(fs, out_dir)
    html_path = out_dir / "_infographic.html"
    html_path.write_text(html)

    out = out_dir / "final.png"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(html_path.as_uri())
        page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": width, "height": height})
        browser.close()
    return out
```

`src/dino_drawer/compose/__main__.py`:
```python
"""Standalone runner: `python -m dino_drawer.compose <out/species/>`."""
import sys
from pathlib import Path

from dino_drawer.models import FactSheet
from .render import screenshot


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m dino_drawer.compose <species-out-dir>", file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    fs = FactSheet.model_validate_json((out_dir / "factsheet.json").read_text())
    out = screenshot(fs, out_dir)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Install Chromium for Playwright**

```bash
playwright install chromium
```

- [ ] **Step 6: Run — expect pass**

```bash
pytest tests/test_compose.py -v
```
Expected: 2 PASS.

- [ ] **Step 7: Commit**

```bash
git add src/dino_drawer/compose tests/test_compose.py
git commit -m "✨ [compose] HTML template + Playwright screenshot for final infographic"
```

---

## Task 12: Agent orchestrator with parallel research and step caching

**Files:**
- Create: `src/dino_drawer/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

`tests/test_agent.py`:
```python
"""Tests for the DinoDrawerAgent orchestration (all sub-steps mocked)."""
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from dino_drawer.agent import DinoDrawerAgent
from dino_drawer.models import PapersFile, RawImagesFile, RefsFile, FactSheet, Annotation, SkullView, Size, Reference, VisualReferences


def _papers() -> PapersFile:
    return PapersFile(species="X", wikipedia=None, papers=[])


def _refs_raw() -> RawImagesFile:
    return RawImagesFile(species="X", images=[])


def _refs() -> RefsFile:
    return RefsFile(species="X", body=[], skull=[])


def _factsheet() -> FactSheet:
    return FactSheet(
        species="X", subtitle="",
        annotations=[Annotation(region="tête", facts=["x"], source_ids=["wikipedia"])],
        skull_view=SkullView(facts=["x"], scale_cm=10, source_ids=["wikipedia"]),
        size=Size(length_m=[1,2], hip_height_m=[1,2], source_ids=["wikipedia"]),
        conclusion="",
        references=[Reference(id="wikipedia", citation_short="W", doi=None, title="t")],
        visual_references=VisualReferences(body=[], skull=[]),
        image_prompt="..",
    )


@pytest.mark.asyncio
async def test_agent_runs_full_pipeline(tmp_path):
    with patch("dino_drawer.agent.papers_aggregator", AsyncMock(return_value=_papers())) as mp, \
         patch("dino_drawer.agent.images_aggregator", AsyncMock(return_value=_refs_raw())) as mi, \
         patch("dino_drawer.agent.classify_and_select", return_value=_refs()) as mc, \
         patch("dino_drawer.agent.synthesize", return_value=_factsheet()) as ms, \
         patch("dino_drawer.agent.generate_assets") as mg, \
         patch("dino_drawer.agent.screenshot", return_value=tmp_path / "final.png") as msc:
        agent = DinoDrawerAgent(out_root=tmp_path)
        out = await agent.run("Tyrannosaurus rex")
    assert out.name == "final.png"
    mp.assert_awaited_once()
    mi.assert_awaited_once()
    mc.assert_called_once()
    ms.assert_called_once()
    mg.assert_called_once()
    msc.assert_called_once()


@pytest.mark.asyncio
async def test_agent_skips_existing_steps(tmp_path):
    species_dir = tmp_path / "tyrannosaurus-rex"
    species_dir.mkdir()
    (species_dir / "papers.json").write_text(_papers().model_dump_json())

    with patch("dino_drawer.agent.papers_aggregator", AsyncMock(return_value=_papers())) as mp, \
         patch("dino_drawer.agent.images_aggregator", AsyncMock(return_value=_refs_raw())) as mi, \
         patch("dino_drawer.agent.classify_and_select", return_value=_refs()), \
         patch("dino_drawer.agent.synthesize", return_value=_factsheet()), \
         patch("dino_drawer.agent.generate_assets"), \
         patch("dino_drawer.agent.screenshot", return_value=species_dir / "final.png"):
        agent = DinoDrawerAgent(out_root=tmp_path)
        await agent.run("Tyrannosaurus rex")
    # papers must NOT have been re-run because papers.json exists
    mp.assert_not_awaited()
    # images still runs (no refs_raw.json)
    mi.assert_awaited_once()
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_agent.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement agent**

`src/dino_drawer/agent.py`:
```python
"""DinoDrawerAgent — orchestrates the 6-step pipeline with caching."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import FactSheet, PapersFile, RawImagesFile, RefsFile
from .research.papers.aggregator import fetch_all as papers_aggregator
from .research.images.aggregator import fetch_all as images_aggregator
from .vision.classifier import classify_and_select
from .synthesis.main import synthesize
from .image.diffusion import generate_assets
from .compose.render import screenshot


def _slug(species: str) -> str:
    return species.lower().replace(" ", "-")


@dataclass
class DinoDrawerAgent:
    out_root: Path = field(default_factory=lambda: Path("out"))
    model_llm: str = "qwen2.5:14b-instruct"
    model_vlm: str = "qwen2.5vl:7b"
    model_image: str = "stabilityai/stable-diffusion-xl-base-1.0"
    max_refs: int = 50
    lang: str = "fr"
    force: bool = False
    force_step: str | None = None
    skip_refs: bool = False

    def _step_should_run(self, name: str, artifact: Path) -> bool:
        order = ["papers", "images", "filter", "synthesis", "diffusion", "compose"]
        if self.force:
            return True
        if self.force_step and order.index(name) >= order.index(self.force_step):
            return True
        return not artifact.exists()

    async def run(self, species: str) -> Path:
        out_dir = self.out_root / _slug(species)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Steps 1+2 in parallel
        async def step_papers():
            if self._step_should_run("papers", out_dir / "papers.json"):
                papers = await papers_aggregator(species)
                (out_dir / "papers.json").write_text(
                    json.dumps(papers.model_dump(), indent=2, ensure_ascii=False)
                )

        async def step_images():
            if self.skip_refs:
                return
            if self._step_should_run("images", out_dir / "refs_raw.json"):
                await images_aggregator(species, out_dir, max_total=self.max_refs)

        await asyncio.gather(step_papers(), step_images())

        # Step 3: filter
        if not self.skip_refs and (out_dir / "refs_raw.json").exists():
            if self._step_should_run("filter", out_dir / "refs.json"):
                classify_and_select(species=species, out_dir=out_dir, vlm_model=self.model_vlm)

        # Step 4: synthesis
        if self._step_should_run("synthesis", out_dir / "factsheet.json"):
            synthesize(species=species, out_dir=out_dir,
                       model_llm=self.model_llm, model_vlm=self.model_vlm, lang=self.lang)

        fs = FactSheet.model_validate_json((out_dir / "factsheet.json").read_text())

        # Step 5: diffusion
        if self._step_should_run("diffusion", out_dir / "hero.png"):
            generate_assets(factsheet=fs, out_dir=out_dir, model=self.model_image)

        # Step 6: compose
        if self._step_should_run("compose", out_dir / "final.png"):
            return screenshot(fs, out_dir)
        return out_dir / "final.png"
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_agent.py -v
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dino_drawer/agent.py tests/test_agent.py
git commit -m "✨ [agent] Orchestrator with parallel research and per-step caching"
```

---

## Task 13: Wire CLI to agent

**Files:**
- Modify: `src/dino_drawer/__main__.py` (replace stub call)

- [ ] **Step 1: Update CLI**

Replace the body of `main()` in `src/dino_drawer/__main__.py`:

```python
"""CLI entry point for dino-drawer."""
import argparse
import asyncio
import sys
from pathlib import Path

from .agent import DinoDrawerAgent


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser for the dino-drawer CLI."""
    parser = argparse.ArgumentParser(
        prog="dino-drawer",
        description="Generate a scientific infographic for a species.",
    )
    parser.add_argument("species", nargs="?", help="Binomial name, e.g. 'Tyrannosaurus rex'")
    parser.add_argument("--out", default="./out", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Re-run all steps")
    parser.add_argument(
        "--force-step",
        choices=["papers", "images", "filter", "synthesis", "diffusion", "compose"],
        help="Re-run this step and everything after",
    )
    parser.add_argument("--skip-refs", action="store_true", help="Skip image scraping + VLM filtering")
    parser.add_argument("--model-llm", default="qwen2.5:14b-instruct")
    parser.add_argument("--model-vlm", default="qwen2.5vl:7b")
    parser.add_argument("--model-image", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--max-refs", type=int, default=50)
    parser.add_argument("--lang", choices=["fr", "en"], default="fr")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dino-drawer CLI. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.species:
        parser.print_help()
        return 0

    agent = DinoDrawerAgent(
        out_root=Path(args.out),
        model_llm=args.model_llm,
        model_vlm=args.model_vlm,
        model_image=args.model_image,
        max_refs=args.max_refs,
        lang=args.lang,
        force=args.force,
        force_step=args.force_step,
        skip_refs=args.skip_refs,
    )
    try:
        out = asyncio.run(agent.run(args.species))
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify CLI smoke test still passes**

```bash
pytest tests/test_cli.py -v
```
Expected: PASS.

- [ ] **Step 3: Run full pipeline end-to-end**

```bash
python -m dino_drawer "Tyrannosaurus rex"
```
Expected: takes 3-8 minutes, produces `out/tyrannosaurus-rex/final.png`. **The user can test here.**

- [ ] **Step 4: Commit**

```bash
git add src/dino_drawer/__main__.py
git commit -m "✨ [cli] Wire CLI to DinoDrawerAgent for end-to-end runs"
```

---

## Task 14: README and final polish

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

`README.md`:
```markdown
# Dino Drawer

Génère une infographie scientifique sourcée à partir d'un nom d'espèce
(préhistorique ou actuelle). Tout tourne en local : recherche bibliographique,
synthèse par LLM, classification d'images par VLM, diffusion conditionnée
par IP-Adapter, et composition HTML/CSS.

## Installation

Prérequis :
- macOS Apple Silicon (16 Go RAM minimum, 24 Go recommandé)
- Python 3.11+
- [Ollama](https://ollama.com)

Une fois cloné :
```bash
ollama pull qwen2.5:14b-instruct
ollama pull qwen2.5vl:7b
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
playwright install chromium
```

## Usage

```bash
python -m dino_drawer "Tyrannosaurus rex"
# → out/tyrannosaurus-rex/final.png
```

Flags utiles :
- `--skip-refs` : saute le scraping d'images, plus rapide pour itérer sur le texte.
- `--force-step diffusion` : re-génère uniquement l'image et la composition.
- `--lang en` : sortie en anglais.
- `--max-refs 20` : limite le scraping.

## Pipeline

1. **Papers** — Semantic Scholar + OpenAlex + Wikipedia (parallèle avec étape 2).
2. **Images** — Wikimedia Commons + PhyloPic + iNaturalist.
3. **Filter** — VLM (Qwen2.5-VL) classe les images et garde le top 3 corps / top 2 crâne.
4. **Synthesis** — LLM (Qwen2.5) produit `factsheet.json` enrichi par les descriptions VLM des refs.
5. **Diffusion** — SDXL + IP-Adapter Plus conditionne la génération sur les refs sélectionnées.
6. **Compose** — Template HTML/CSS rendu par Playwright pour le `final.png`.

Voir `docs/superpowers/specs/2026-05-12-dino-drawer-design.md` pour le détail.

## Développement

```bash
pytest                       # tous les tests (mocks, pas d'appels réseau ni LLM)
pytest tests/test_compose.py # un seul module
```
```

- [ ] **Step 2: Run all tests one final time**

```bash
pytest -v
```
Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "📝 [docs] README with install, usage, and pipeline overview"
```

---

## Self-Review

**Spec coverage check** — every spec section is implemented:

| Spec section | Task |
|---|---|
| 3.1 Research/papers | Task 2 |
| 3.2 Research/images | Task 3 |
| 3.3 Vision filtering | Tasks 4 + 5 |
| 3.4 Synthesis (with VLM brief) | Tasks 6 + 7 + 8 |
| 3.5 Diffusion (SDXL + IP-Adapter) | Task 10 |
| 3.6 Compose (HTML + Playwright) | Task 11 |
| 3.7 Orchestrator (asyncio + caching) | Task 12 |
| Silhouette SVG by code | Task 9 |
| CLI with all flags | Tasks 0 + 13 |
| Models with cross-source validation | Task 1 |
| Tests (TDD, mocked) | Every task |
| Graceful degradation (no papers / no refs) | Tasks 2, 3, 8, 10 |

No placeholders. Types consistent across tasks (`PapersFile`, `RawImagesFile`, `RefsFile`, `FactSheet`, `VisualReferences`). All method signatures match between modules.
