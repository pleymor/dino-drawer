"""Build the per-species entry and maintain the local ``published/catalog.json``.

Layout chosen for the static-site workflow:

- Only the final ``.webp`` image is uploaded to R2 (one file per species).
- A single ``published/catalog.json`` lives in the repo and is committed.
  It contains every species' full meta plus the R2 image URL.

The catalog dict has the shape::

    {
        "generated_at": "...",
        "count": N,
        "species": [<species_entry>, ...]
    }

A ``species_entry`` is the factsheet content plus ``image_url`` and ``generated_at``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from dino_drawer.models import FactSheet


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with seconds precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_species_entry(factsheet: FactSheet, image_url: str) -> dict:
    """Combine factsheet text with the R2 image URL into one species entry.

    Args:
        factsheet: Validated FactSheet for the species.
        image_url: Public R2 URL of the uploaded final ``.webp``.

    Returns:
        Plain dict suitable for inclusion in ``catalog.json``.
    """
    return {
        "slug": factsheet.species.lower().replace(" ", "-"),
        "species": factsheet.species,
        "subtitle": factsheet.subtitle,
        "annotations": [a.model_dump() for a in factsheet.annotations],
        "skull_view": factsheet.skull_view.model_dump(),
        "size": factsheet.size.model_dump(),
        "conclusion": factsheet.conclusion,
        "references": [r.model_dump() for r in factsheet.references],
        "image_prompt": factsheet.image_prompt,
        "image_url": image_url,
        "generated_at": _now_iso(),
    }


def upsert_catalog(existing: dict | None, entry: dict) -> dict:
    """Insert or replace ``entry`` in the catalog, sorted by species name."""
    if existing is None:
        existing = {"species": []}
    species_list = [s for s in existing.get("species", []) if s["slug"] != entry["slug"]]
    species_list.append(entry)
    species_list.sort(key=lambda s: s["species"])
    return {
        "generated_at": _now_iso(),
        "count": len(species_list),
        "species": species_list,
    }


def remove_from_catalog(existing: dict, slug: str) -> dict:
    """Remove the entry matching ``slug`` from the catalog."""
    species_list = [s for s in existing.get("species", []) if s["slug"] != slug]
    return {
        "generated_at": _now_iso(),
        "count": len(species_list),
        "species": species_list,
    }
