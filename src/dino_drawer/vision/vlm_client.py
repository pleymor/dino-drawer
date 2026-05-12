"""Thin wrapper around Ollama's vision API with JSON validation + retry."""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import ollama
from PIL import Image

_MAX_SIDE_PX = 512


class VLMError(RuntimeError):
    """Raised when the VLM cannot produce valid JSON after retries."""


_CLASSIFY_PROMPT = """\
Tu analyses une image candidate pour servir de référence à la génération
d'illustration scientifique de {species}.

Réponds en JSON strict, sans markdown :
{{
  "type": "paleoart_realiste" | "rendu_3d" | "photo_squelette" | "fossile" | "schema_anatomique" | "cladogramme" | "illustration_enfant" | "photo_specimen_vivant" | "carte_distribution" | "autre",
  "view": "profil_corps" | "trois_quarts_corps" | "face_corps" | "crane_profil" | "crane_face" | "detail" | "scene_groupe" | "autre",
  "usable_for_body_generation": true|false,
  "usable_for_skull_generation": true|false,
  "realism_score": 0-10,
  "quality_score": 0-10,
  "description_courte": "string"
}}"""


def _downscaled_bytes(image_path: Path, max_side: int = _MAX_SIDE_PX) -> bytes:
    """Return JPEG bytes of the image, downscaled so longest side <= max_side.

    Large images (1920×1280+) produce too many vision tokens for the
    Ollama context window and slow inference dramatically. Downscaling
    to ~512px gives the VLM enough detail for classification while
    keeping inference fast.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _extract_json(text: str) -> dict | None:
    """Try to parse JSON from a text that may contain markdown fences or prose.

    Args:
        text: Raw text potentially containing JSON.

    Returns:
        Parsed JSON dict, or None if parsing failed.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class VLMClient:
    """Ollama vision client with JSON validation and retry-on-invalid."""

    def __init__(self, model: str = "qwen2.5vl:7b", max_retries: int = 2) -> None:
        """Initialize the VLM client.

        Args:
            model: Ollama model name (default qwen2.5vl:7b).
            max_retries: Number of retries on invalid JSON (default 2).
        """
        self.model = model
        self.max_retries = max_retries

    def classify_image(self, image_path: Path, species: str) -> dict:
        """Classify a single image. Returns dict matching the JSON schema.

        Args:
            image_path: Path to image file.
            species: Scientific name of the species.

        Returns:
            Parsed JSON dict with image classification metadata.

        Raises:
            VLMError: If VLM cannot produce valid JSON after max_retries.
        """
        prompt = _CLASSIFY_PROMPT.format(species=species)
        small = _downscaled_bytes(Path(image_path))
        last_err = ""
        for attempt in range(self.max_retries + 1):
            user_content = prompt if attempt == 0 else (
                f"{prompt}\n\nPrécédente tentative invalide ({last_err}). Renvoie uniquement le JSON."
            )
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": user_content, "images": [small]}],
                options={"temperature": 0.0},
            )
            content = resp.get("message", {}).get("content", "")
            parsed = _extract_json(content)
            if parsed is not None and "type" in parsed:
                return parsed
            last_err = "missing 'type' or non-JSON output"
        raise VLMError(f"VLM failed to return valid JSON after {self.max_retries + 1} tries")

    def describe_image(self, image_path: Path, prompt: str) -> str:
        """Free-form description of an image for prompt-enrichment.

        Args:
            image_path: Path to image file.
            prompt: Descriptive prompt for the model.

        Returns:
            Model response text.
        """
        small = _downscaled_bytes(Path(image_path))
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [small]}],
            options={"temperature": 0.2},
        )
        return resp.get("message", {}).get("content", "").strip()
