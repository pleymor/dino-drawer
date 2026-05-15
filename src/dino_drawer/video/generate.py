"""Image-to-video generation via Veo 3.1.

Reads ``hero.png`` (and ``factsheet.json``) from a species output directory and
produces ``hero.mp4`` next to them. The image anchors the scene; a short motion
prompt derived from the factsheet animates the dinosaur.

The motion prompt is composed in three layers, mirroring ``image/diffusion.py``:

  [_UNIVERSAL_MOTION] + [_CLADE_MOTION[factsheet.clade]] + [species-specific]

* Universal layer: camera discipline (locked-off witness shot), continuity with
  the still, ambient micro-motion (foliage, dust, light).
* Clade layer: motion arc + species-appropriate behaviour (a charging theropod,
  a high-browsing sauropod, an alert ceratopsian, …).
* Species layer: pulled from ``factsheet.image_prompt`` so the environment and
  integument stay consistent with the still frame.

Adding a new clade = one entry in ``_CLADE_MOTION``.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

from dino_drawer.models import FactSheet

_DEFAULT_MODEL = "veo-3.1-fast-generate-preview"


def _video_model() -> str:
    return os.environ.get("GEMINI_VIDEO_MODEL", _DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Universal motion preamble — short on purpose. Veo 3.1-lite hallucinates when
# the prompt drowns in scene-setting; the input image already carries lighting,
# environment, anatomy and integument. We only describe camera + motion arc.
# ---------------------------------------------------------------------------

_UNIVERSAL_MOTION = (
    "Live-action wildlife footage continuing directly from the input still "
    "image — same subject, same lighting, same colour palette, same depth of "
    "field, same vegetation. The camera is LOCKED-OFF on a tripod: no pan, no "
    "tilt, no zoom, no dolly, no cut, no scene change. The framing stays "
    "constant for the full duration. "
)

_NO_TEXT_MOTION_FOOTER = (
    "No text, no captions, no watermark, no logo, no UI overlay."
)


# ---------------------------------------------------------------------------
# Clade-specific motion blocks.
# ---------------------------------------------------------------------------

_CLADE_THEROPODA = (
    # Action arc — plain language, no timing buckets (Veo ignores them).
    "The dinosaur in the image suddenly charges forward and runs straight at "
    "the camera. Bipedal sprint: long powerful hind-limb strides, body held "
    "horizontal, tail extended behind as a counter-balance — NEVER an upright "
    "kangaroo posture. The head grows in the frame as the animal closes the "
    "distance. At the closest point it throws its head toward the lens and "
    "lets out a deep impressive roar aimed at the camera. "
    # Lipped anatomy when the jaws open — short, sharp, anti-grin.
    "When the jaws open for the roar, the scaly lips stay continuous over the "
    "upper and lower marginal teeth the way a modern monitor lizard's lips do "
    "when it gapes — only the INNER mouth (tongue, palate, gum line) is "
    "visible. Absolutely NO bared side teeth, NO wall of fangs, NO Jurassic-"
    "Park grin, NO hyena-style lip retraction. Saliva strings; warm breath "
    "vapour catches the light. "
    # Physics.
    "Heavy multi-tonne biped: each hind-foot landing shakes the ground, dust "
    "and debris kick up behind the feet, ferns whip aside as the body brushes "
    "past, the tail tip whips on each stride. Real mass and inertia — never "
    "floaty. "
)

_CLADE_OTHER = (
    "The dinosaur in the image performs a calm, characteristic behaviour "
    "appropriate to its biology: small head movements, slow weight shifts, "
    "occasional tail or neck motion, visible breathing at the flanks. No "
    "dramatic action, no scene change. "
)

_CLADE_MOTION: dict[str, str] = {
    "theropoda": _CLADE_THEROPODA,
    # Other clades fall back to _CLADE_OTHER until tuned.
}


def _motion_prompt(factsheet: FactSheet) -> str:
    """Compose the Veo prompt: universal camera rules + clade-specific arc.

    Intentionally does NOT append ``factsheet.image_prompt``: that prompt was
    designed for a still frame (scene-setting, integument, body length) and
    fights with the input image when fed to Veo, which causes hallucinations.
    """
    clade = (factsheet.clade or "other").strip().lower()
    clade_block = _CLADE_MOTION.get(clade, _CLADE_OTHER)
    return f"{_UNIVERSAL_MOTION}{clade_block}{_NO_TEXT_MOTION_FOOTER}"


def generate_video(
    factsheet: FactSheet,
    out_dir: Path,
    *,
    model: str | None = None,
    duration_seconds: int = 8,
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
) -> Path:
    """Generate ``hero.mp4`` in *out_dir* from ``hero.png`` + factsheet.

    Args:
        factsheet: Validated FactSheet (used for ``species``, ``clade`` and
            ``image_prompt``).
        out_dir: Species output directory; must contain ``hero.png``.
        model: Veo model name. None uses ``GEMINI_VIDEO_MODEL`` or the default.
        duration_seconds: 5-8.
        aspect_ratio: ``16:9`` or ``16:10``.
        resolution: ``720p``, ``1080p`` or ``4k``.

    Returns:
        The path to the written ``hero.mp4``.
    """
    out_dir = Path(out_dir)
    hero_png = out_dir / "hero.png"
    if not hero_png.exists():
        raise FileNotFoundError(
            f"{hero_png} not found — run the image step first (`make image`)."
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")

    client = genai.Client(
        http_options={"api_version": "v1beta"},
        api_key=api_key,
    )

    config = types.GenerateVideosConfig(
        # Veo 3.1-lite rejects "dont_allow"; "allow_adult" is the safe default
        # since the prompt never depicts humans.
        person_generation="allow_adult",
        aspect_ratio=aspect_ratio,
        number_of_videos=1,
        duration_seconds=duration_seconds,
        resolution=resolution,
    )

    source = types.GenerateVideosSource(
        prompt=_motion_prompt(factsheet),
        image=types.Image(
            image_bytes=hero_png.read_bytes(),
            mime_type="image/png",
        ),
    )

    effective_model = model or _video_model()
    operation = client.models.generate_videos(
        model=effective_model,
        source=source,
        config=config,
    )

    while not operation.done:
        print("Veo: video not ready yet, polling again in 10 s…", flush=True)
        time.sleep(10)
        operation = client.operations.get(operation)

    result = operation.result
    if not result or not result.generated_videos:
        raise RuntimeError("Veo returned no videos.")

    generated = result.generated_videos[0]
    client.files.download(file=generated.video)
    out_mp4 = out_dir / "hero.mp4"
    generated.video.save(str(out_mp4))

    # Record which model produced this video so publish/catalog can show provenance.
    meta = {
        "video_model": effective_model,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "video_meta.json").write_text(json.dumps(meta, indent=2))
    return out_mp4
