"""CLI entry point for the publish pipeline.

Usage
-----
Publish a species::

    .venv/bin/python -m dino_drawer.publish <slug-or-out-path>

Remove a species from R2 and the catalog::

    .venv/bin/python -m dino_drawer.publish <slug-or-out-path> --unpublish
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dino_drawer.models import FactSheet
from dino_drawer.publish.manifest import build_meta, remove_from_catalog, upsert_catalog
from dino_drawer.publish.optimize import optimize_image
from dino_drawer.publish.r2 import R2Client, R2Error

_CATALOG_KEY = "catalog.json"


def _resolve_out_dir(target: str) -> tuple[Path, str]:
    """Return ``(out_dir, slug)`` from a slug string or an ``out/<slug>`` path.

    Parameters
    ----------
    target:
        Either a slug such as ``tyrannosaurus-rex`` or a path like
        ``out/tyrannosaurus-rex``.
    """
    p = Path(target)
    if p.exists() and p.is_dir():
        return p, p.name
    # Treat as raw slug — look relative to current working directory
    candidate = Path("out") / target
    if candidate.exists():
        return candidate, target
    raise SystemExit(f"Cannot find output directory for {target!r}")


def _load_factsheet(out_dir: Path) -> FactSheet:
    """Load and parse ``factsheet.json`` from *out_dir*.

    Parameters
    ----------
    out_dir:
        The species output directory containing ``factsheet.json``.
    """
    path = out_dir / "factsheet.json"
    if not path.exists():
        raise SystemExit(f"factsheet.json not found in {out_dir}")
    return FactSheet.model_validate_json(path.read_text())


def _fetch_catalog(client: R2Client) -> dict | None:
    """Fetch the current ``catalog.json`` from R2, or ``None`` if absent.

    Parameters
    ----------
    client:
        Authenticated :class:`~dino_drawer.publish.r2.R2Client` instance.
    """
    raw = client.get_bytes(_CATALOG_KEY)
    if raw is None:
        return None
    return json.loads(raw)


def _push_catalog(client: R2Client, catalog: dict) -> str:
    """Serialise and upload the catalog dict to R2.

    Parameters
    ----------
    client:
        Authenticated :class:`~dino_drawer.publish.r2.R2Client` instance.
    catalog:
        Catalog dict to upload.

    Returns
    -------
    str
        Public URL of the uploaded ``catalog.json``.
    """
    return client.upload_bytes(
        json.dumps(catalog, ensure_ascii=False, indent=2).encode(),
        _CATALOG_KEY,
        content_type="application/json",
    )


def _publish(out_dir: Path, slug: str, client: R2Client) -> None:
    """Optimise images, upload all assets to R2, and refresh the catalog.

    Parameters
    ----------
    out_dir:
        Species output directory (must contain ``hero.png`` and ``skull.png``).
    slug:
        URL-safe species identifier, e.g. ``tyrannosaurus-rex``.
    client:
        Authenticated :class:`~dino_drawer.publish.r2.R2Client` instance.
    """
    for name in ("hero.png", "skull.png"):
        if not (out_dir / name).exists():
            raise SystemExit(f"Required file missing: {out_dir / name}")

    factsheet = _load_factsheet(out_dir)
    publish_dir = out_dir / "_publish"

    image_urls: dict[str, dict[int, str]] = {}
    for kind in ("hero", "skull"):
        src = out_dir / f"{kind}.png"
        print(f"  Optimising {src.name} …")
        webp_paths = optimize_image(src, publish_dir, kind)
        urls: dict[int, str] = {}
        for w, webp_path in webp_paths.items():
            key = f"{slug}/{kind}@{w}.webp"
            print(f"  Uploading {key} …")
            url = client.upload_file(webp_path, key, content_type="image/webp")
            urls[w] = url
            print(f"    → {url}")
        image_urls[kind] = urls

    meta = build_meta(factsheet, image_urls)
    meta_key = f"{slug}/meta.json"
    meta_url = client.upload_bytes(
        json.dumps(meta, ensure_ascii=False, indent=2).encode(),
        meta_key,
        content_type="application/json",
    )
    print(f"  Uploaded meta → {meta_url}")

    catalog = _fetch_catalog(client)
    catalog = upsert_catalog(catalog, meta)
    catalog_url = _push_catalog(client, catalog)
    print(f"  Catalog updated → {catalog_url}")


def _unpublish(slug: str, client: R2Client) -> None:
    """Delete all R2 objects for *slug* and remove it from the catalog.

    Parameters
    ----------
    slug:
        URL-safe species identifier, e.g. ``tyrannosaurus-rex``.
    client:
        Authenticated :class:`~dino_drawer.publish.r2.R2Client` instance.
    """
    prefix = f"{slug}/"
    print(f"  Deleting R2 objects under {prefix!r} …")
    count = client.delete_prefix(prefix)
    print(f"  Deleted {count} object(s).")

    catalog = _fetch_catalog(client)
    if catalog is not None:
        catalog = remove_from_catalog(catalog, slug)
        catalog_url = _push_catalog(client, catalog)
        print(f"  Catalog updated → {catalog_url}")
    else:
        print("  No catalog found; nothing to update.")


def main() -> None:
    """Parse arguments and dispatch to publish or unpublish."""
    parser = argparse.ArgumentParser(
        prog="python -m dino_drawer.publish",
        description="Publish or unpublish a dino-drawer species to Cloudflare R2.",
    )
    parser.add_argument(
        "target",
        help="Slug or path to the species output directory (e.g. out/tyrannosaurus-rex).",
    )
    parser.add_argument(
        "--unpublish",
        action="store_true",
        default=False,
        help="Remove the species from R2 and the catalog.",
    )
    args = parser.parse_args()

    out_dir, slug = _resolve_out_dir(args.target)
    print(f"{'Unpublishing' if args.unpublish else 'Publishing'} {slug!r} …")

    try:
        client = R2Client()
    except R2Error as exc:
        raise SystemExit(str(exc)) from exc

    if args.unpublish:
        _unpublish(slug, client)
    else:
        _publish(out_dir, slug, client)

    print("Done.")


if __name__ == "__main__":
    main()
