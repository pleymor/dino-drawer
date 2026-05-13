"""Thin wrapper around Gemini for image classification and description."""
from __future__ import annotations

from pathlib import Path

from dino_drawer.clients.gemini import GeminiClient, GeminiError


class VLMError(RuntimeError):
    """Raised when the VLM cannot produce a usable result."""


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


class VLMClient:
    """Gemini-backed vision client with JSON validation and retry-on-invalid."""

    def __init__(self, model: str | None = None, max_retries: int = 2) -> None:
        """Initialize. `model` overrides GEMINI_TEXT_MODEL env default.

        Args:
            model: Gemini model name. Defaults to None (uses GEMINI_TEXT_MODEL
                env var or ``gemini-2.5-flash``).
            max_retries: Number of retries on invalid JSON responses (default 2).
                Network retries are handled internally by GeminiClient.
        """
        self.model = model
        self.max_retries = max_retries
        self._client = GeminiClient()

    def classify_image(self, image_path: Path, species: str) -> dict:
        """Classify a single image. Returns the parsed JSON dict.

        Sends the image to Gemini with a structured classification prompt and
        retries if the response dict is missing the required ``type`` key.

        Args:
            image_path: Path to the image file.
            species: Scientific name of the species being classified.

        Returns:
            Parsed JSON dict matching the classification schema.

        Raises:
            VLMError: If Gemini fails or returns invalid classification after
                all retries are exhausted.
        """
        prompt = _CLASSIFY_PROMPT.format(species=species)
        last_err = ""
        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    data = self._client.chat_json(prompt, images=[image_path], model=self.model)
                else:
                    retry_prompt = (
                        f"{prompt}\n\nPrécédente tentative invalide ({last_err}). "
                        "Renvoie uniquement le JSON."
                    )
                    data = self._client.chat_json(retry_prompt, images=[image_path], model=self.model)
            except GeminiError as exc:
                raise VLMError(f"Gemini call failed: {exc}") from exc
            if isinstance(data, dict) and "type" in data:
                return data
            last_err = "missing 'type' key"
        raise VLMError(f"VLM failed to return valid classification after {self.max_retries + 1} tries")

    def describe_image(self, image_path: Path, prompt: str) -> str:
        """Return free-form text description of the image.

        Args:
            image_path: Path to the image file.
            prompt: Descriptive instruction for the model.

        Returns:
            Stripped text response from Gemini.

        Raises:
            VLMError: If the Gemini API call fails.
        """
        try:
            return self._client.chat(prompt, images=[image_path], model=self.model).strip()
        except GeminiError as exc:
            raise VLMError(f"Gemini call failed: {exc}") from exc
