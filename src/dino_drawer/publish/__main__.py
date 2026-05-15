"""CLI entry point for the publish pipeline.

Uploads the species' final image (as WebP) to Cloudflare R2 and refreshes the
local ``published/catalog.json`` that the static site consumes.

Usage::

    .venv/bin/python -m dino_drawer.publish <slug-or-out-path>
    .venv/bin/python -m dino_drawer.publish <slug-or-out-path> --unpublish
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from dino_drawer.models import FactSheet
from dino_drawer.publish.manifest import (
    build_species_entry,
    remove_from_catalog,
    upsert_catalog,
)
from dino_drawer.publish.r2 import R2Client, R2Error

_CATALOG_PATH = Path("published/catalog.json")
_WEBP_QUALITY = 85
_THUMBNAIL_MAX_DIM = 400
_THUMBNAIL_QUALITY = 80


def _resolve_out_dir(target: str) -> tuple[Path, str]:
    """Return ``(out_dir, slug)`` from a slug string or an ``out/<slug>`` path."""
    p = Path(target)
    if p.exists() and p.is_dir():
        return p, p.name
    candidate = Path("out") / target
    if candidate.exists():
        return candidate, target
    raise SystemExit(f"Cannot find output directory for {target!r}")


def _load_factsheet(out_dir: Path) -> FactSheet:
    """Load and parse ``factsheet.json`` from *out_dir*."""
    path = out_dir / "factsheet.json"
    if not path.exists():
        raise SystemExit(f"factsheet.json not found in {out_dir}")
    return FactSheet.model_validate_json(path.read_text())


def _png_to_webp(src_png: Path, dst_webp: Path) -> None:
    """Convert *src_png* to WebP at its native resolution, quality 85."""
    img = Image.open(src_png).convert("RGB")
    dst_webp.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_webp, "WEBP", quality=_WEBP_QUALITY, method=6)


def _png_to_webp_thumbnail(src_png: Path, dst_webp: Path) -> None:
    """Convert *src_png* to a downscaled WebP thumbnail.

    Long edge is capped at ``_THUMBNAIL_MAX_DIM`` px; aspect ratio is preserved.
    """
    img = Image.open(src_png).convert("RGB")
    img.thumbnail((_THUMBNAIL_MAX_DIM, _THUMBNAIL_MAX_DIM), Image.Resampling.LANCZOS)
    dst_webp.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_webp, "WEBP", quality=_THUMBNAIL_QUALITY, method=6)


def _load_local_catalog() -> dict | None:
    """Read the local ``published/catalog.json`` if it exists."""
    if not _CATALOG_PATH.exists():
        return None
    return json.loads(_CATALOG_PATH.read_text())


def _read_image_model(out_dir: Path) -> str:
    """Read ``image_meta.json`` written by diffusion. Falls back to 'unknown'."""
    meta_path = out_dir / "image_meta.json"
    if not meta_path.exists():
        return "unknown"
    try:
        return json.loads(meta_path.read_text()).get("image_model", "unknown")
    except (json.JSONDecodeError, OSError):
        return "unknown"


def _read_video_model(out_dir: Path) -> str:
    """Read ``video_meta.json`` written by the video step. Falls back to 'unknown'."""
    meta_path = out_dir / "video_meta.json"
    if not meta_path.exists():
        return "unknown"
    try:
        return json.loads(meta_path.read_text()).get("video_model", "unknown")
    except (json.JSONDecodeError, OSError):
        return "unknown"


def _write_local_catalog(catalog: dict) -> None:
    """Write the catalog dict to ``published/catalog.json``."""
    _CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2))


def _publish(out_dir: Path, slug: str, client: R2Client) -> None:
    """Optimise the hero image, upload to R2, refresh the local catalog."""
    hero_png = out_dir / "hero.png"
    if not hero_png.exists():
        raise SystemExit(f"Required file missing: {hero_png}")

    factsheet = _load_factsheet(out_dir)

    webp = out_dir / "_publish" / f"{slug}.webp"
    print(f"  Converting {hero_png.name} → {webp.name} …")
    _png_to_webp(hero_png, webp)

    thumb_webp = out_dir / "_publish" / f"{slug}_thumbnail.webp"
    print(f"  Generating thumbnail → {thumb_webp.name} (max {_THUMBNAIL_MAX_DIM}px) …")
    _png_to_webp_thumbnail(hero_png, thumb_webp)

    key = f"{slug}.webp"
    print(f"  Uploading {key} → R2 …")
    image_url = client.upload_file(webp, key, content_type="image/webp")
    print(f"    → {image_url}")

    thumb_key = f"{slug}_thumbnail.webp"
    print(f"  Uploading {thumb_key} → R2 …")
    thumbnail_url = client.upload_file(thumb_webp, thumb_key, content_type="image/webp")
    print(f"    → {thumbnail_url}")

    # Optional video — upload hero.mp4 if present.
    video_url: str | None = None
    video_model: str | None = None
    hero_mp4 = out_dir / "hero.mp4"
    if hero_mp4.exists():
        mp4_key = f"{slug}.mp4"
        print(f"  Uploading {mp4_key} → R2 …")
        video_url = client.upload_file(hero_mp4, mp4_key, content_type="video/mp4")
        video_model = _read_video_model(out_dir)
        print(f"    → {video_url}")

    image_model = _read_image_model(out_dir)
    entry = build_species_entry(
        factsheet,
        image_url,
        thumbnail_url,
        image_model=image_model,
        video_url=video_url,
        video_model=video_model,
    )
    catalog = upsert_catalog(_load_local_catalog(), entry)
    _write_local_catalog(catalog)
    video_info = f", video_model={video_model}" if video_url else ""
    print(
        f"  Catalog updated → {_CATALOG_PATH} ({catalog['count']} species, "
        f"image_model={image_model}{video_info})"
    )


def _unpublish(slug: str, client: R2Client) -> None:
    """Delete the slug's image + thumbnail from R2 and remove from the local catalog."""
    total = 0
    for key in (f"{slug}.webp", f"{slug}_thumbnail.webp", f"{slug}.mp4"):
        print(f"  Deleting {key} from R2 …")
        total += client.delete_prefix(key)
    print(f"  Deleted {total} object(s).")

    catalog = _load_local_catalog()
    if catalog is None:
        print("  No local catalog found; nothing to update.")
        return
    catalog = remove_from_catalog(catalog, slug)
    _write_local_catalog(catalog)
    print(f"  Catalog updated → {_CATALOG_PATH} ({catalog['count']} species)")


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
        help="Remove the species from R2 and the local catalog.",
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
