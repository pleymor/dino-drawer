"""CLI entry point for dino-drawer."""
import argparse
import asyncio
import sys
from pathlib import Path

from .agent import DinoDrawerAgent


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser for the dino-drawer CLI."""
    parser = argparse.ArgumentParser(
        prog="dino-drawer",
        description="Generate a scientific infographic for a species.",
    )
    parser.add_argument("species", nargs="?", help="Binomial name, e.g. 'Tyrannosaurus rex'")
    parser.add_argument("--out", default="./out", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Re-run all steps")
    parser.add_argument(
        "--force-step",
        choices=["papers", "images", "filter", "synthesis", "diffusion"],
        help="Re-run this step and everything after",
    )
    parser.add_argument("--skip-refs", action="store_true", help="Skip image scraping + VLM filtering")
    parser.add_argument("--model-llm", default=None,
                        help="Gemini text model (default: $GEMINI_TEXT_MODEL or gemini-2.5-flash)")
    parser.add_argument("--model-vlm", default=None,
                        help="Gemini multimodal model (default: same as --model-llm)")
    parser.add_argument("--model-image", default=None,
                        help="Gemini image model (default: $GEMINI_IMAGE_MODEL or gemini-3-pro-image-preview)")
    parser.add_argument("--max-refs", type=int, default=50)
    parser.add_argument("--lang", choices=["fr", "en"], default="fr")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dino-drawer CLI. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.species:
        parser.print_help()
        return 0

    agent = DinoDrawerAgent(
        out_root=Path(args.out),
        model_llm=args.model_llm,
        model_vlm=args.model_vlm,
        model_image=args.model_image,
        max_refs=args.max_refs,
        lang=args.lang,
        force=args.force,
        force_step=args.force_step,
        skip_refs=args.skip_refs,
    )
    try:
        out = asyncio.run(agent.run(args.species))
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
