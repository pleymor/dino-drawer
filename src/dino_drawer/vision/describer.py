"""Generate a `visual_brief` paragraph from selected body references."""
from pathlib import Path

from dino_drawer.models import RefsFile
from .vlm_client import VLMClient

_DESCRIBE_PROMPT = (
    "Describe this image in 2 sentences: skin colours and patterns, "
    "presence and location of feathers/hair/scales, posture, visible environment. "
    "Do not name the species."
)


def describe_body_refs(
    refs: RefsFile, out_dir: Path, vlm_model: str = "qwen2.5vl:7b"
) -> str:
    """Call the VLM on each body ref, return a single concatenated brief string.

    Processes each body reference image through the VLM using a standard prompt
    to generate descriptions of skin color, patterns, feathers, posture, and
    environment. The descriptions are concatenated into a single visual brief.

    Args:
        refs: RefsFile containing classified body references to describe.
        out_dir: Output directory where reference images are stored (relative paths
            in refs.body are resolved from this directory).
        vlm_model: Ollama model name to use for vision tasks (default "qwen2.5vl:7b").

    Returns:
        Concatenated description string with one description per line, or empty string
        if no usable body references exist.
    """
    vlm = VLMClient(model=vlm_model)
    parts: list[str] = []
    for c in refs.body:
        img_path = Path(out_dir) / c.path
        if not img_path.exists():
            continue
        parts.append(vlm.describe_image(img_path, _DESCRIBE_PROMPT))
    return "\n".join(parts)
