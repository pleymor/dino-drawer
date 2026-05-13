"""Gemini-backed image generation. Produces hero.png, skull.png, silhouette.svg."""
from __future__ import annotations

from pathlib import Path

from dino_drawer.clients.gemini import GeminiClient
from dino_drawer.models import FactSheet
from .silhouette import build_scale_svg


def _load_ref_paths(out_dir: Path, paths: list[str]) -> list[Path]:
    """Resolve relative ref paths against out_dir; skip any missing files.

    Args:
        out_dir: Base directory that paths are relative to.
        paths: List of relative file paths.

    Returns:
        List of absolute Path objects for files that exist on disk.
    """
    out: list[Path] = []
    for p in paths:
        full = out_dir / p
        if full.exists():
            out.append(full)
    return out


def _hero_prompt(factsheet: FactSheet) -> str:
    """Return the prompt for the hero illustration.

    Wraps the LLM-generated image_prompt with photographic-realism cues
    and explicit no-text directives.
    """
    base = factsheet.image_prompt.strip().rstrip(".")
    return (
        "Unedited handheld wildlife photograph, RAW DSLR image, National Geographic style, "
        "shot on full-frame 35mm prime lens, subtle ISO grain, natural lens distortion, "
        "slightly imperfect framing as if captured by a witness one second before being attacked. "
        "ABSOLUTELY NOT an illustration, drawing, painting, render, CGI, 3D model, paleoart, "
        "or studio shot. "
        f"{base}. "
        "Framing: front-facing or near head-on, the animal is charging straight toward the camera, "
        "head thrust forward and pointed at the viewer, three-quarter front view, "
        "entire body in frame — head, body, both legs, and most of the tail visible, "
        "no cropping of the head or shoulders. "
        "Pose: aggressive charge, muscles bunched, claws splayed, body weight on the front leg, "
        "low menacing glare locked on the camera, neck feathers raised. "
        "Anatomical accuracy: mouth firmly shut, lip line forms a continuous scaly seam "
        "from the snout tip to the jaw corner — like a closed-mouth monitor lizard or "
        "a resting Komodo dragon — with ABSOLUTELY ZERO teeth visible, not even a tip, "
        "no fangs protruding, the upper lip overhangs and conceals the entire dental row. "
        "ABSOLUTELY NO TEXT anywhere — no captions, no watermark, no signature, no UI overlays, no logo, no labels."
    )


def _skull_prompt(factsheet: FactSheet) -> str:
    """Return the prompt for the skull side-view.

    Args:
        factsheet: Species factsheet providing the species name.

    Returns:
        Full prompt string for the skull generation call.
    """
    return (
        f"Hyper-realistic museum photograph of a {factsheet.species} skull fossil, "
        "lateral profile, soft directional lighting, neutral dark background, "
        "fine detail on bone texture, anatomically accurate, shot on 50mm prime lens. "
        "No text, no captions, no labels, no watermark, no signature."
    )


def generate_assets(
    factsheet: FactSheet,
    out_dir: Path,
    *,
    model: str | None = None,
    steps: int | None = None,
) -> None:
    """Produce hero.png, skull.png, silhouette.svg in out_dir via Gemini.

    Args:
        factsheet: Fully validated FactSheet with image_prompt and visual_references.
        out_dir: Directory where outputs are written.
        model: Gemini image model name. None uses GEMINI_IMAGE_MODEL env default.
        steps: Ignored — kept for signature compatibility.
    """
    del steps  # unused with Gemini backend
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiClient()

    body_refs = _load_ref_paths(out_dir, [r.path for r in factsheet.visual_references.body])
    skull_refs = _load_ref_paths(out_dir, [r.path for r in factsheet.visual_references.skull])

    hero_bytes = client.generate_image(
        _hero_prompt(factsheet),
        refs=body_refs,
        model=model,
    )
    (out_dir / "hero.png").write_bytes(hero_bytes)

    skull_bytes = client.generate_image(
        _skull_prompt(factsheet),
        refs=skull_refs,
        model=model,
    )
    (out_dir / "skull.png").write_bytes(skull_bytes)

    length = sum(factsheet.size.length_m) / len(factsheet.size.length_m)
    hip = sum(factsheet.size.hip_height_m) / len(factsheet.size.hip_height_m)
    (out_dir / "silhouette.svg").write_text(build_scale_svg(length_m=length, hip_height_m=hip))
