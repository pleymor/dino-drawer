"""SDXL + IP-Adapter pipeline. Generates hero.png, skull.png, silhouette.svg."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from dino_drawer.models import FactSheet
from .silhouette import build_scale_svg

_NEGATIVE = "text, watermark, captions, signature, logo, multiple animals, deformed anatomy"


def _load_pipeline(model: str):
    """Lazy-load the SDXL pipeline with IP-Adapter weights on MPS.

    Separated so tests can monkey-patch it.

    Args:
        model: HuggingFace model identifier for the SDXL base checkpoint.

    Returns:
        A loaded StableDiffusionXLPipeline with IP-Adapter weights attached.
    """
    import torch
    from diffusers import StableDiffusionXLPipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        image_encoder_folder="models/image_encoder",
    )
    pipe.set_ip_adapter_scale(0.6)
    return pipe


def _load_ref_images(out_dir: Path, paths: list[str]) -> list[Image.Image]:
    """Load reference images from disk relative to out_dir.

    Args:
        out_dir: Base directory that paths are relative to.
        paths: List of relative file paths to load.

    Returns:
        List of RGB PIL images.
    """
    return [Image.open(out_dir / p).convert("RGB") for p in paths]


def _build_hero_kwargs(
    factsheet: FactSheet,
    body_refs: list[Image.Image],
    steps: int,
) -> dict:
    """Assemble keyword arguments for the hero image pipeline call.

    Args:
        factsheet: Species factsheet containing the image prompt.
        body_refs: Loaded body reference images; may be empty.
        steps: Number of diffusion inference steps.

    Returns:
        Keyword argument dict ready to unpack into the pipeline.
    """
    kwargs: dict = dict(
        prompt=factsheet.image_prompt,
        negative_prompt=_NEGATIVE,
        num_inference_steps=steps,
        width=1024,
        height=640,
    )
    if body_refs:
        # Wrap as list-of-lists: outer length = number of loaded IP-Adapters (1),
        # inner length = number of reference images for that adapter.
        kwargs["ip_adapter_image"] = [body_refs]
    return kwargs


def _build_skull_kwargs(
    factsheet: FactSheet,
    skull_refs: list[Image.Image],
    steps: int,
) -> dict:
    """Assemble keyword arguments for the skull image pipeline call.

    Args:
        factsheet: Species factsheet used to build the skull prompt.
        skull_refs: Loaded skull reference images; may be empty.
        steps: Number of diffusion inference steps.

    Returns:
        Keyword argument dict ready to unpack into the pipeline.
    """
    skull_prompt = (
        f"detailed lateral view of {factsheet.species} skull, scientific illustration style, "
        "neutral background, no text"
    )
    kwargs: dict = dict(
        prompt=skull_prompt,
        negative_prompt=_NEGATIVE,
        num_inference_steps=steps,
        width=640,
        height=480,
    )
    if skull_refs:
        kwargs["ip_adapter_image"] = [skull_refs]
    return kwargs


def generate_assets(
    factsheet: FactSheet,
    out_dir: Path,
    *,
    model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    steps: int = 30,
) -> None:
    """Produce hero.png, skull.png, silhouette.svg in out_dir.

    Loads the SDXL pipeline (via :func:`_load_pipeline`) and runs two
    inference passes — hero and skull — each optionally conditioned on
    visual reference images via IP-Adapter when refs are available.
    The silhouette SVG is generated from size data without GPU work.

    Args:
        factsheet: Fully validated :class:`~dino_drawer.models.FactSheet`.
        out_dir: Directory where output files are written.
        model: HuggingFace model ID for the SDXL base checkpoint.
        steps: Number of diffusion inference steps (lower = faster, lower quality).
    """
    out_dir = Path(out_dir)
    pipe = _load_pipeline(model)

    body_refs = _load_ref_images(out_dir, [r.path for r in factsheet.visual_references.body])
    skull_refs = _load_ref_images(out_dir, [r.path for r in factsheet.visual_references.skull])

    skull_out = pipe(**_build_skull_kwargs(factsheet, skull_refs, steps))
    skull_out.images[0].save(out_dir / "skull.png")

    hero_out = pipe(**_build_hero_kwargs(factsheet, body_refs, steps))
    hero_out.images[0].save(out_dir / "hero.png")

    length = sum(factsheet.size.length_m) / len(factsheet.size.length_m)
    hip = sum(factsheet.size.hip_height_m) / len(factsheet.size.hip_height_m)
    svg = build_scale_svg(length_m=length, hip_height_m=hip)
    (out_dir / "silhouette.svg").write_text(svg)
