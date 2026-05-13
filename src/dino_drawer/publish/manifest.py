"""Build per-species ``meta.json`` and the global ``catalog.json``."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from dino_drawer.models import FactSheet


def build_meta(factsheet: FactSheet, image_urls: dict[str, dict[int, str]]) -> dict:
    """Combine *factsheet* text with R2 image URLs into a serialisable meta dict.

    Parameters
    ----------
    factsheet:
        Validated ``FactSheet`` for the species.
    image_urls:
        Mapping of image kind → width → public URL.
        Example: ``{"hero": {1600: "https://…/hero@1600.webp", …}, "skull": {…}}``.

    Returns
    -------
    dict
        A plain dict suitable for JSON serialisation and upload as ``meta.json``.
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
        "images": {
            kind: {str(w): url for w, url in widths.items()}
            for kind, widths in image_urls.items()
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def upsert_catalog(existing: dict | None, meta: dict) -> dict:
    """Insert or update *meta* in the global catalog dict.

    The catalog has the shape::

        {
            "generated_at": "…",
            "count": N,
            "species": [
                {"slug": …, "species": …, "subtitle": …,
                 "thumb_url": …, "meta_url": …, "generated_at": …},
                …
            ]
        }

    Species entries are kept sorted alphabetically by ``species`` name.

    Parameters
    ----------
    existing:
        Current catalog dict fetched from R2, or ``None`` if none exists yet.
    meta:
        The species meta dict produced by :func:`build_meta`.

    Returns
    -------
    dict
        Updated catalog dict ready for JSON serialisation.
    """
    if existing is None:
        existing = {"species": []}
    species_list = existing.get("species", [])
    # Remove stale entry for this slug (if any)
    species_list = [s for s in species_list if s["slug"] != meta["slug"]]
    smallest_hero = meta["images"]["hero"].get("400") or next(
        iter(meta["images"]["hero"].values())
    )
    species_list.append(
        {
            "slug": meta["slug"],
            "species": meta["species"],
            "subtitle": meta["subtitle"],
            "thumb_url": smallest_hero,
            "meta_url": f"{smallest_hero.rsplit('/', 1)[0]}/meta.json",
            "generated_at": meta["generated_at"],
        }
    )
    species_list.sort(key=lambda s: s["species"])
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(species_list),
        "species": species_list,
    }


def remove_from_catalog(existing: dict, slug: str) -> dict:
    """Remove the entry with *slug* from the catalog.

    Parameters
    ----------
    existing:
        Current catalog dict fetched from R2.
    slug:
        Species slug to remove, e.g. ``"tyrannosaurus-rex"``.

    Returns
    -------
    dict
        Updated catalog dict with the species removed.
    """
    species_list = [s for s in existing.get("species", []) if s["slug"] != slug]
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(species_list),
        "species": species_list,
    }
