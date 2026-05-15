"""Central Gemini API client with retry, JSON mode, and image generation.

Wraps the official ``google-genai`` SDK (>=1.0) and provides three high-level
methods that handle rate-limit / transient errors via exponential backoff.

Environment variables
---------------------
GEMINI_API_KEY      — required, loaded automatically via python-dotenv.
GEMINI_TEXT_MODEL   — default text model (default: ``gemini-2.5-flash``).
GEMINI_IMAGE_MODEL  — default image model (default: ``gemini-3-pro-image-preview``).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from google import genai
from google.api_core import exceptions as gexc
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_TEXT_MODEL = "gemini-2.5-flash"
_DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
_MAX_RETRIES = 8
_MAX_DELAY_SECONDS = 60

# Errors that warrant a retry (rate-limit / transient server errors).
_RETRYABLE = (
    gexc.ResourceExhausted,  # 429
    gexc.ServiceUnavailable,  # 503
    gexc.InternalServerError,  # 500
    gexc.DeadlineExceeded,  # 504
)

# Errors that should surface immediately as GeminiError without retrying.
_PERMANENT = (
    gexc.PermissionDenied,   # 403 — bad API key / no access
    gexc.Unauthenticated,    # 401
    gexc.InvalidArgument,    # 400
    gexc.NotFound,           # 404
)


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------

class GeminiError(RuntimeError):
    """Raised on permanent Gemini API failures or exhausted retries.

    Attributes:
        cause: The original exception that triggered this error, if any.
    """

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        """Initialise GeminiError.

        Args:
            message: Human-readable error description.
            cause: Underlying exception, if available.
        """
        super().__init__(message)
        self.cause = cause


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _text_model() -> str:
    """Return the configured default text model name."""
    return os.environ.get("GEMINI_TEXT_MODEL", _DEFAULT_TEXT_MODEL)


def _image_model() -> str:
    """Return the configured default image model name."""
    return os.environ.get("GEMINI_IMAGE_MODEL", _DEFAULT_IMAGE_MODEL)


def _call_with_retry(fn, *args, **kwargs):
    """Call *fn* with *args/kwargs*, retrying on transient errors.

    Uses exponential backoff starting at 1 s, doubling each attempt (1, 2, 4, 8 …).
    Raises immediately on permanent errors. After *_MAX_RETRIES* retries, wraps
    the last retryable error in a :class:`GeminiError`.

    Args:
        fn: Callable to invoke.
        *args: Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn*.

    Raises:
        GeminiError: On permanent errors or after max retries.
    """
    delay = 1
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except _PERMANENT as exc:
            _raise_permanent(exc)
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            wait = min(_extract_retry_delay(exc) or delay, _MAX_DELAY_SECONDS)
            time.sleep(wait)
            delay = min(delay * 2, _MAX_DELAY_SECONDS)
        except genai_errors.ClientError as exc:
            # google-genai raises this for 4xx; treat 429 as retryable, others permanent.
            if exc.code == 429:
                last_exc = exc
                if attempt == _MAX_RETRIES:
                    break
                wait = min(_extract_retry_delay(exc) or delay, _MAX_DELAY_SECONDS)
                time.sleep(wait)
                delay = min(delay * 2, _MAX_DELAY_SECONDS)
            else:
                _raise_permanent(exc)
        except genai_errors.ServerError as exc:
            # 5xx: always retryable.
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            wait = min(_extract_retry_delay(exc) or delay, _MAX_DELAY_SECONDS)
            time.sleep(wait)
            delay = min(delay * 2, _MAX_DELAY_SECONDS)
        except Exception as exc:
            raise GeminiError(f"Unexpected Gemini API error: {exc}", cause=exc) from exc

    raise GeminiError(
        f"Gemini rate-limit / ResourceExhausted after {_MAX_RETRIES} retries: {last_exc}",
        cause=last_exc,
    ) from last_exc


def _extract_retry_delay(exc: Exception) -> float | None:
    """Parse `Please retry in Xs` or RetryInfo.retry_delay from a Gemini error.

    Gemini's 429 / 503 responses often include a precise server-side hint about
    how long to wait. Honouring it converges faster than blind exponential
    backoff. Returns *None* when no hint is available.
    """
    import re

    msg = str(exc)
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"'retryDelay':\s*'([0-9]+(?:\.[0-9]+)?)s'", msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _raise_permanent(exc: Exception) -> None:
    """Convert a permanent API exception into a :class:`GeminiError`.

    Args:
        exc: The original google-api-core exception.

    Raises:
        GeminiError: Always.
    """
    if isinstance(exc, (gexc.PermissionDenied, gexc.Unauthenticated)):
        raise GeminiError(
            f"Gemini auth / API key error — check GEMINI_API_KEY: {exc}", cause=exc
        ) from exc
    raise GeminiError(f"Gemini permanent API error: {exc}", cause=exc) from exc


def _build_contents(prompt: str, images: list[Path | bytes]) -> list:
    """Assemble a contents list from a text prompt and optional images.

    Args:
        prompt: Text instruction.
        images: Sequence of image paths or raw bytes.

    Returns:
        List suitable for ``client.models.generate_content(contents=...)``.
    """
    from PIL import Image
    import io

    contents: list = [prompt]
    for img in images:
        if isinstance(img, (str, Path)):
            contents.append(Image.open(Path(img)))
        else:
            contents.append(Image.open(io.BytesIO(img)))
    return contents


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------

class GeminiClient:
    """Thin wrapper around the google-genai SDK.

    Provides three high-level methods (``chat``, ``chat_json``,
    ``generate_image``) with automatic exponential-backoff retry on
    rate-limit and transient server errors.

    Args:
        api_key: Gemini API key. Defaults to the ``GEMINI_API_KEY``
            environment variable (auto-loaded via python-dotenv).
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the Gemini client.

        Args:
            api_key: Optional API key override. Falls back to
                ``GEMINI_API_KEY`` env var if not supplied.

        Raises:
            GeminiError: If no API key can be found.
        """
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise GeminiError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or pass api_key= explicitly."
            )
        self._client = genai.Client(api_key=key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        images: list[Path | bytes] | None = None,
        *,
        model: str | None = None,
    ) -> str:
        """Send a free-form text (or multimodal) prompt and return the response.

        Args:
            prompt: Instruction to the model.
            images: Optional images to include (file paths or raw bytes).
            model: Override the model. Defaults to ``GEMINI_TEXT_MODEL``
                env var or ``gemini-2.5-flash``.

        Returns:
            The model's text response.

        Raises:
            GeminiError: On auth failures, quota exhaustion, or bad requests.
        """
        model = model or _text_model()
        contents = _build_contents(prompt, images or [])

        resp = _call_with_retry(
            self._client.models.generate_content,
            model=model,
            contents=contents,
        )
        return resp.text

    def chat_json(
        self,
        prompt: str,
        images: list[Path | bytes] | None = None,
        *,
        model: str | None = None,
    ) -> dict:
        """Send a prompt requesting a JSON response and parse it.

        Uses ``response_mime_type="application/json"`` to instruct the model
        to return valid JSON. Falls back to regex extraction if the model wraps
        its output in markdown. Retries up to *_MAX_RETRIES* times on parse
        failures before raising.

        Args:
            prompt: Instruction to the model.
            images: Optional images to include.
            model: Override the model. Defaults to ``GEMINI_TEXT_MODEL``
                env var or ``gemini-2.5-flash``.

        Returns:
            Parsed JSON as a ``dict``.

        Raises:
            GeminiError: If the response cannot be parsed as JSON after retries,
                or on any permanent / exhausted API error.
        """
        model = model or _text_model()
        contents = _build_contents(prompt, images or [])
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
        )

        last_text = ""
        for attempt in range(_MAX_RETRIES + 1):
            resp = _call_with_retry(
                self._client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
            last_text = resp.text or ""
            parsed = _try_parse_json(last_text)
            if parsed is not None:
                return parsed
            if attempt < _MAX_RETRIES:
                time.sleep(1)

        raise GeminiError(
            f"Failed to parse JSON from Gemini response after {_MAX_RETRIES + 1} "
            f"attempts. Last response: {last_text!r}"
        )

    def generate_image(
        self,
        prompt: str,
        refs: list[Path | bytes] | None = None,
        *,
        model: str | None = None,
    ) -> bytes:
        """Generate an image from a text prompt and optional reference images.

        Calls the Gemini image-generation endpoint and extracts the raw PNG
        bytes from the first ``inline_data`` part in the response.

        Args:
            prompt: Image description / instruction.
            refs: Optional reference images (file paths or raw bytes).
            model: Override the model. Defaults to ``GEMINI_IMAGE_MODEL``
                env var or ``gemini-3-pro-image-preview``.

        Returns:
            Raw PNG bytes of the generated image.

        Raises:
            GeminiError: If no image is found in the response, or on API errors.
        """
        model = model or _image_model()
        contents = _build_contents(prompt, refs or [])

        resp = _call_with_retry(
            self._client.models.generate_content,
            model=model,
            contents=contents,
        )

        for part in resp.candidates[0].content.parts:
            if part.inline_data:
                return part.inline_data.data

        raise GeminiError(
            "No image inline_data found in Gemini response. "
            "The model may not have generated an image."
        )


# ---------------------------------------------------------------------------
# Internal JSON helpers
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON from *text*, handling markdown fences.

    Args:
        text: Raw text that may contain JSON or markdown-fenced JSON.

    Returns:
        Parsed dict, or ``None`` if parsing fails.
    """
    import re

    text = text.strip()
    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    # Last-resort: grab the first {...} block.
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        try:
            result = json.loads(brace.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return None
