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
    n_skull: int = 2,
) -> RefsFile:
    """Read refs_raw.json, classify with VLM, select top-N body and skull refs.

    Writes refs.json and copies kept images to refs/.

    Args:
        species: Scientific name of the species.
        out_dir: Directory containing refs_raw.json and refs_raw/ images.
        vlm_model: Ollama model to use for classification.
        n_body: Maximum number of body reference images to keep.
        n_skull: Maximum number of skull reference images to keep.

    Returns:
        RefsFile with selected body and skull images.
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

    def _copy_and_rewrite(items: list[ClassifiedImage], prefix: str) -> list[ClassifiedImage]:
        """Copy selected images into refs/ and update their paths.

        Args:
            items: Classified images to copy.
            prefix: Filename prefix (``body`` or ``skull``).

        Returns:
            New list of ClassifiedImage with updated paths under refs/.
        """
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
