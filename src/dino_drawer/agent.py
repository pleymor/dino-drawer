"""DinoDrawerAgent — orchestrates the 6-step pipeline with caching.

Each step writes a JSON/PNG artifact to ``<out_root>/<slug>/``.  A step is
skipped when its artifact already exists, unless *force* is set or
*force_step* points to an earlier stage.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import FactSheet, PapersFile, RawImagesFile, RefsFile
from .research.papers.aggregator import fetch_all as papers_aggregator
from .research.images.aggregator import fetch_all as images_aggregator
from .vision.classifier import classify_and_select
from .synthesis.main import synthesize
from .image.diffusion import generate_assets
from .compose.render import screenshot

#: Canonical pipeline order used for *force_step* comparisons.
_STEP_ORDER = ["papers", "images", "filter", "synthesis", "diffusion", "compose"]


def _slug(species: str) -> str:
    """Convert a species name to a filesystem-safe slug.

    Examples::

        >>> _slug("Tyrannosaurus rex")
        'tyrannosaurus-rex'
    """
    return species.lower().replace(" ", "-")


@dataclass
class DinoDrawerAgent:
    """Orchestrator that runs the dino-drawer pipeline end-to-end.

    Parameters
    ----------
    out_root:
        Root output directory.  Each species is written to
        ``<out_root>/<slug>/``.
    model_llm:
        Ollama model tag used for text synthesis.
    model_vlm:
        Ollama model tag used for vision classification/description.
    model_image:
        Diffusion model ID used for image generation.
    max_refs:
        Maximum number of reference images to download.
    lang:
        Target language for the generated fact-sheet (ISO-639-1 code).
    force:
        When *True*, re-run every step regardless of cached artifacts.
    force_step:
        Re-run this step and all subsequent ones (e.g. ``"synthesis"``).
    skip_refs:
        When *True*, skip the images + filter steps entirely.
    """

    out_root: Path = field(default_factory=lambda: Path("out"))
    model_llm: str | None = None
    model_vlm: str | None = None
    model_image: str | None = None
    max_refs: int = 50
    lang: str = "fr"
    force: bool = False
    force_step: str | None = None
    skip_refs: bool = False

    def _step_should_run(self, name: str, artifact: Path) -> bool:
        """Return *True* when *name* step must be executed.

        A step runs when any of the following is true:

        * ``force`` is set.
        * ``force_step`` names a step that is at the same position or earlier
          in the pipeline order.
        * The expected *artifact* does not yet exist on disk.

        Parameters
        ----------
        name:
            Pipeline step identifier (must be in ``_STEP_ORDER``).
        artifact:
            File that the step is expected to produce.
        """
        if self.force:
            return True
        if self.force_step and _STEP_ORDER.index(name) >= _STEP_ORDER.index(self.force_step):
            return True
        return not artifact.exists()

    async def run(self, species: str) -> Path:
        """Execute the full dino-drawer pipeline for *species*.

        The six steps run in order:

        1. **papers** — aggregate research papers (parallel with images).
        2. **images** — download reference photos (parallel with papers).
        3. **filter** — classify and select reference images.
        4. **synthesis** — build the fact-sheet from papers + descriptions.
        5. **diffusion** — generate the hero illustration.
        6. **compose** — render the final infographic and take a screenshot.

        Parameters
        ----------
        species:
            Scientific species name, e.g. ``"Tyrannosaurus rex"``.

        Returns
        -------
        Path
            Path to the final rendered PNG (``<out_dir>/final.png``).
        """
        out_dir = self.out_root / _slug(species)
        out_dir.mkdir(parents=True, exist_ok=True)

        async def step_papers() -> None:
            """Fetch and persist research papers for *species*."""
            if self._step_should_run("papers", out_dir / "papers.json"):
                papers = await papers_aggregator(species)
                (out_dir / "papers.json").write_text(
                    json.dumps(papers.model_dump(), indent=2, ensure_ascii=False)
                )

        images_ran: bool = False

        async def step_images() -> None:
            """Download and persist reference images for *species*."""
            nonlocal images_ran
            if self.skip_refs:
                return
            if self._step_should_run("images", out_dir / "refs_raw.json"):
                await images_aggregator(species, out_dir, max_total=self.max_refs)
                images_ran = True

        # Steps 1 & 2 are independent — run in parallel.
        await asyncio.gather(step_papers(), step_images())

        # Step 3: filter reference images.
        # Run when refs exist on disk (previous run) or were just fetched now.
        refs_available = images_ran or (out_dir / "refs_raw.json").exists()
        if not self.skip_refs and refs_available:
            if self._step_should_run("filter", out_dir / "refs.json"):
                classify_and_select(species=species, out_dir=out_dir, vlm_model=self.model_vlm)

        # Step 4: synthesize fact-sheet.
        if self._step_should_run("synthesis", out_dir / "factsheet.json"):
            fs = synthesize(
                species=species,
                out_dir=out_dir,
                model_llm=self.model_llm,
                model_vlm=self.model_vlm,
                lang=self.lang,
            )
        else:
            fs = FactSheet.model_validate_json((out_dir / "factsheet.json").read_text())

        # Step 5: generate hero illustration.
        if self._step_should_run("diffusion", out_dir / "hero.png"):
            generate_assets(factsheet=fs, out_dir=out_dir, model=self.model_image)

        # Step 6: render final infographic.
        # Playwright's sync API doesn't tolerate a running asyncio loop, so we
        # run the screenshot call in a worker thread.
        if self._step_should_run("compose", out_dir / "final.png"):
            return await asyncio.to_thread(screenshot, fs, out_dir)
        return out_dir / "final.png"
