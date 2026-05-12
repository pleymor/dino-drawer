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
