"""Smoke test for the CLI entry point."""
import subprocess
import sys


def test_cli_help_returns_zero():
    """`python -m dino_drawer --help` must exit 0 and print usage."""
    result = subprocess.run(
        [sys.executable, "-m", "dino_drawer", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "species" in result.stdout.lower() or "usage" in result.stdout.lower()
