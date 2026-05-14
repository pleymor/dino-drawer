"""Apply VLM classification to raw images, then select best per usage."""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from dino_drawer.models import (
    ClassifiedImage,
    RawImage,
    RawImagesFile,
    RefsFile,
)
from .vlm_client import VLMClient, VLMError

_REJECTED_TYPES = {"cladogramme", "illustration_enfant", "carte_distribution"}
_BODY_TYPES = {"paleoart_realiste", "rendu_3d", "photo_specimen_vivant"}
_BODY_VIEWS = {"profil_corps", "trois_quarts_corps"}

# Concurrent VLM classification calls. Tuned for Gemini's default per-minute
# request limits; raise if you have a higher quota.
_VLM_CONCURRENCY = 4


def _score(c: ClassifiedImage) -> float:
    """Compute a composite score for ranking classified images.

    Args:
        c: A classified image with realism and quality scores.

    Returns:
        Product of realism_score and quality_score.
    """
    return c.realism_score * c.quality_score


def classify_and_select(
    species: str,
    out_dir: Path,
    *,
    vlm_model: str = "qwen2.5vl:7b",
    n_body: int = 3,
) -> RefsFile:
    """Read refs_raw.json, classify with VLM, select top-N body refs.

    Writes refs.json and copies kept body images to refs/.

    Args:
        species: Scientific name of the species.
        out_dir: Directory containing refs_raw.json and refs_raw/ images.
        vlm_model: Ollama model to use for classification.
        n_body: Maximum number of body reference images to keep.

    Returns:
        RefsFile with selected body images.
    """
    out_dir = Path(out_dir)
    raw_meta = RawImagesFile.model_validate_json((out_dir / "refs_raw.json").read_text())
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "classifications_cache.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    vlm = VLMClient(model=vlm_model)
    total = len(raw_meta.images)

    # First pass: figure out what still needs to be classified. Anything with
    # a missing file or already in the cache is recorded and skipped here.
    pending: list[tuple[int, RawImage]] = []
    missing_count = 0
    for idx, raw in enumerate(raw_meta.images, 1):
        img_path = out_dir / raw.path
        if not img_path.exists():
            print(f"[{idx}/{total}] skip (missing) id={raw.id}", flush=True)
            missing_count += 1
            continue
        cache_key = str(raw.id)
        if cache_key in cache:
            print(
                f"[{idx}/{total}] cached id={raw.id} "
                f"type={cache[cache_key].get('type', '?')}",
                flush=True,
            )
        else:
            pending.append((idx, raw))

    # Classify pending images concurrently. The cache is updated incrementally
    # under a lock so a crash mid-run preserves the in-progress work.
    cache_lock = Lock()
    vlm_failures = 0
    failures_lock = Lock()

    def _classify_one(idx_raw: tuple[int, RawImage]) -> None:
        nonlocal vlm_failures
        idx, raw = idx_raw
        img_path = out_dir / raw.path
        print(f"[{idx}/{total}] classifying id={raw.id} ({raw.path})", flush=True)
        try:
            data = vlm.classify_image(img_path, species=species)
        except VLMError as exc:
            print(f"[{idx}/{total}]   VLM error: {exc}", flush=True)
            with failures_lock:
                vlm_failures += 1
            return
        with cache_lock:
            cache[str(raw.id)] = data
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

    if pending:
        with ThreadPoolExecutor(max_workers=_VLM_CONCURRENCY) as ex:
            for _ in ex.map(_classify_one, pending):
                pass

    # Second pass: build the classified list from the (now complete) cache.
    classified: list[ClassifiedImage] = []
    rejected_types_count = 0
    for raw in raw_meta.images:
        img_path = out_dir / raw.path
        if not img_path.exists():
            continue  # counted in missing_count
        cache_key = str(raw.id)
        if cache_key not in cache:
            continue  # counted in vlm_failures
        data = cache[cache_key]
        if data.get("type") in _REJECTED_TYPES:
            rejected_types_count += 1
            continue
        classified.append(ClassifiedImage(
            id=raw.id,
            path=raw.path,
            type=data["type"],
            view=data["view"],
            usable_for_body_generation=bool(data.get("usable_for_body_generation")),
            realism_score=float(data.get("realism_score", 0)),
            quality_score=float(data.get("quality_score", 0)),
            description_courte=data.get("description_courte", ""),
            credit=raw.credit,
            license=raw.license,
        ))

    rejected = missing_count + vlm_failures + rejected_types_count

    body_candidates = [c for c in classified
                       if c.usable_for_body_generation
                       and c.type in _BODY_TYPES
                       and c.view in _BODY_VIEWS
                       and c.realism_score >= 6]
    body_top = sorted(body_candidates, key=_score, reverse=True)[:n_body]

    def _copy_and_rewrite(items: list[ClassifiedImage]) -> list[ClassifiedImage]:
        """Copy selected body images into refs/ and update their paths."""
        out: list[ClassifiedImage] = []
        for i, c in enumerate(items):
            src = out_dir / c.path
            ext = src.suffix
            dst_rel = f"refs/body_{i}{ext}"
            shutil.copy(src, out_dir / dst_rel)
            out.append(c.model_copy(update={"path": dst_rel}))
        return out

    body_top = _copy_and_rewrite(body_top)

    refs = RefsFile(
        species=species,
        body=body_top,
        rejected_count=rejected + (len(classified) - len(body_top)),
    )
    (out_dir / "refs.json").write_text(json.dumps(refs.model_dump(), indent=2, ensure_ascii=False))
    return refs
