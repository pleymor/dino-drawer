"""Smoke-test Imagen 4 Ultra on an existing factsheet's hero prompt.

Compares against the Gemini Image hero.png already in out/<slug>/.

Usage:
    .venv/bin/python scripts/test_imagen4.py <slug>            # default 1:1 aspect
    .venv/bin/python scripts/test_imagen4.py <slug> 16:9       # specific aspect
    .venv/bin/python scripts/test_imagen4.py <slug> 16:9 ultra # ultra (default) or standard

Writes ``out/<slug>/hero_imagen4.png`` for visual comparison.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from dino_drawer.image.diffusion import _hero_prompt
from dino_drawer.models import FactSheet


_MODELS = {
    "ultra":    "imagen-4.0-ultra-generate-001",
    "standard": "imagen-4.0-generate-001",
    "fast":     "imagen-4.0-fast-generate-001",
}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: test_imagen4.py <slug> [aspect] [tier]", file=sys.stderr)
        print("       aspect: 1:1 (default) | 16:9 | 9:16 | 4:3 | 3:4", file=sys.stderr)
        print("       tier  : ultra (default) | standard | fast", file=sys.stderr)
        return 1
    slug = sys.argv[1]
    aspect = sys.argv[2] if len(sys.argv) > 2 else "1:1"
    tier = sys.argv[3] if len(sys.argv) > 3 else "ultra"
    model_id = _MODELS[tier]

    out_dir = Path("out") / slug
    factsheet_path = out_dir / "factsheet.json"
    if not factsheet_path.exists():
        print(f"factsheet not found: {factsheet_path}", file=sys.stderr)
        return 1

    fs = FactSheet.model_validate_json(factsheet_path.read_text())
    prompt = _hero_prompt(fs)

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY missing in .env", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)

    print(f"==> species  : {fs.species}")
    print(f"==> clade    : {fs.clade}")
    print(f"==> model    : {model_id}")
    print(f"==> aspect   : {aspect}")
    print(f"==> prompt len: {len(prompt)} chars")
    print("==> calling Imagen 4 …", flush=True)

    response = client.models.generate_images(
        model=model_id,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=aspect,
        ),
    )

    if not response.generated_images:
        print("ERROR: no image returned", file=sys.stderr)
        # Try to surface why
        for attr in ("positive_prompt_safety_attributes",
                     "rai_filtered_reason", "filtered_reason"):
            val = getattr(response, attr, None)
            if val:
                print(f"  {attr}: {val}", file=sys.stderr)
        print(f"  raw response: {response!r}"[:2000], file=sys.stderr)
        return 2
    image_bytes = response.generated_images[0].image.image_bytes

    suffix = f"_imagen4-{tier}-{aspect.replace(':', 'x')}"
    out_path = out_dir / f"hero{suffix}.png"
    out_path.write_bytes(image_bytes)
    print(f"==> wrote {out_path} ({len(image_bytes)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
