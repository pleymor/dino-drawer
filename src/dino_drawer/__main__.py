"""CLI entry point for dino-drawer."""
import argparse
import sys


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
        choices=["papers", "images", "filter", "synthesis", "diffusion", "compose"],
        help="Re-run this step and everything after",
    )
    parser.add_argument("--skip-refs", action="store_true", help="Skip image scraping + VLM filtering")
    parser.add_argument("--model-llm", default="qwen2.5:14b-instruct")
    parser.add_argument("--model-vlm", default="qwen2.5vl:7b")
    parser.add_argument("--model-image", default="stabilityai/stable-diffusion-xl-base-1.0")
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
    print(f"[stub] would generate infographic for {args.species!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
