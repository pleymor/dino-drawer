"""Gemini-backed hero image generation. Produces hero.png."""
from __future__ import annotations

from pathlib import Path

from dino_drawer.clients.gemini import GeminiClient
from dino_drawer.models import FactSheet


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


_COMMON_HERO_PROMPT = (
    # 1. Medium & style — anchor the photo aesthetic.
    "Unedited handheld wildlife photograph, RAW DSLR image, National Geographic style, "
    "shot on full-frame 35mm prime lens, subtle ISO grain, natural lens distortion, "
    "framing slightly imperfect as if captured by a witness one second before being attacked. "
    "ABSOLUTELY NOT an illustration, drawing, painting, render, CGI, 3D model, paleoart, "
    "or studio shot. "
    # 2. Framing + pose — charging head-on, but mouth nearly closed (not a wide snarl).
    "The animal is moving briskly toward the camera in a three-quarter front view, "
    "head thrust forward and pointed at the viewer, muscles bunched, claws splayed, "
    "body weight on the front leg, focused predatory stare locked on the camera. "
    "Mouth only slightly parted — NOT a wide-open snarling roar, NOT a gaping jaw; "
    "lips remain compressed, jaw barely separated. "
    "Entire body in frame — head, body, both legs and the full tail visible, "
    "no cropping of the head or shoulders; animal occupies 60-80% of the frame. "
    # 3. Theropod posture — positive framing only (image models amplify mentioned concepts,
    #    so we drop the "NOT upright/NOT kangaroo" negations).
    "Body held horizontally, spine roughly parallel to the ground, "
    "massive head and long muscular tail balance each other around the hips like a seesaw, "
    "tail extended horizontally behind the body and clearly visible from base to tip, "
    "the head-to-tail line is essentially level. "
    # 4. Theropod skull anatomy — Sue/Stan-grade.
    "Skull anatomy based on modern CT scans of the most complete specimens: "
    "forward-facing eyes producing strong binocular overlap, external nares on the rostrum "
    "near the snout tip, deep boxy skull profile. "
    # 4b. Theropod lips (Cullen et al. 2023) — scaly extra-oral tissue fully covers teeth.
    "Theropod lips: the skull is sheathed in scaly extra-oral tissue (lips) that FULLY "
    "covers the dental row, like a modern monitor lizard. Because the mouth is only "
    "slightly parted (not snarling), the lips stay in contact along most of their length; "
    "at most the very tips of TWO OR THREE front teeth peek out at the front of the upper "
    "and lower jaw — the rest of the dental row stays completely hidden behind the lip "
    "seal. The lip line forms a continuous scaly seam from the snout tip to the jaw "
    "corner. NEVER show the side teeth (the long curved fangs behind the canines must "
    "stay hidden); NEVER show teeth root-to-tip; NEVER show a wide grin like a stylized "
    "Jurassic Park T. rex. "
    # 4c. Dental anatomy — strict rules to prevent common image-gen failures.
    "Dental anatomy: the upper teeth are SHORT relative to the snout depth and DO NOT "
    "extend below the lower jaw lip line — they stay above it, even when the mouth is "
    "slightly parted. The front of both jaws has a COMPLETE row of teeth (the four "
    "premaxillary teeth at the very tip of the upper jaw are present, the front dentary "
    "teeth on the lower jaw are present); NEVER depict gaps, missing front teeth, or a "
    "toothless snout tip. If teeth are visible at all, they appear as small uniform tips "
    "emerging just above the lip line at the front — not as oversized, downward-pointing "
    "fangs hanging past the lower jaw. "
    # 5. Camera/light.
    "Eye-level, shallow depth of field, golden hour or overcast soft light. "
)

_NO_TEXT_FOOTER = (
    "ABSOLUTELY NO TEXT anywhere — no captions, no watermark, no signature, "
    "no UI overlays, no logo, no labels."
)


def _hero_prompt(factsheet: FactSheet) -> str:
    """Return the hero prompt: common universal directives + species-specific content.

    ``factsheet.image_prompt`` is expected to contain ONLY the species-specific
    content (paleoenvironment, tegument, palette, unique features, size).
    Universal style/framing/pose/anatomy directives are prepended from
    ``_COMMON_HERO_PROMPT`` and the no-text footer is appended.
    """
    species_specific = factsheet.image_prompt.strip().rstrip(".")
    return f"{_COMMON_HERO_PROMPT}{species_specific}. {_NO_TEXT_FOOTER}"


def generate_assets(
    factsheet: FactSheet,
    out_dir: Path,
    *,
    model: str | None = None,
    steps: int | None = None,
) -> None:
    """Produce ``hero.png`` in ``out_dir`` via Gemini.

    Args:
        factsheet: Fully validated FactSheet with image_prompt and visual_references.
        out_dir: Directory where ``hero.png`` is written.
        model: Gemini image model name. None uses GEMINI_IMAGE_MODEL env default.
        steps: Ignored — kept for signature compatibility.
    """
    del steps  # unused with Gemini backend
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiClient()

    body_refs = _load_ref_paths(out_dir, [r.path for r in factsheet.visual_references.body])
    hero_bytes = client.generate_image(
        _hero_prompt(factsheet), refs=body_refs, model=model
    )
    (out_dir / "hero.png").write_bytes(hero_bytes)
