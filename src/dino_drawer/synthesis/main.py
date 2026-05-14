"""Top-level synthesis: papers + refs + visual_brief -> factsheet.json."""
from __future__ import annotations

import json
from pathlib import Path

from dino_drawer.models import (
    FactSheet,
    PapersFile,
    RefsFile,
    VisualRef,
    VisualReferences,
)
from dino_drawer.vision.describer import describe_body_refs
from .ollama_client import call_llm_for_json
from .prompts import build_synthesis_prompt


def _load_refs(out_dir: Path) -> RefsFile | None:
    """
    Load and validate the refs.json file from the output directory.

    Args:
        out_dir: Directory containing refs.json.

    Returns:
        Validated RefsFile, or None if refs.json does not exist.
    """
    p = out_dir / "refs.json"
    if not p.exists():
        return None
    return RefsFile.model_validate_json(p.read_text())


def _sanitize_source_ids(raw: dict) -> None:
    """Drop or remap source_ids that don't exist in the references list.

    LLMs occasionally hallucinate paper indices (e.g. citing ``paper:14``
    when only ``paper:0..paper:9`` are in references). Rather than failing
    validation, we replace unknown ids with a fallback ("wikipedia" if
    present, otherwise the first known id). The dict is mutated in place.
    """
    known = {r["id"] for r in raw.get("references", [])}
    fallback = "wikipedia" if "wikipedia" in known else (next(iter(known), None))
    if fallback is None:
        return  # no references at all — let validation surface that

    def _fix(ids: list[str]) -> list[str]:
        return [sid if sid in known else fallback for sid in ids]

    for block_name in ("dimensions", "integument", "posture", "habitat", "signature_traits"):
        block = raw.get(block_name)
        if isinstance(block, dict) and "source_ids" in block:
            block["source_ids"] = _fix(block.get("source_ids", []))


def _build_visual_references(refs: RefsFile | None) -> VisualReferences:
    """
    Convert classified body images from a RefsFile into VisualReferences.

    Computes a composite score as realism_score * quality_score / 10 for
    each image.

    Args:
        refs: Validated RefsFile, or None if no refs are available.

    Returns:
        VisualReferences with the body list populated (empty if refs is None).
    """
    if refs is None:
        return VisualReferences(body=[])
    return VisualReferences(
        body=[
            VisualRef(
                path=c.path,
                credit=c.credit,
                license=c.license,
                score=c.realism_score * c.quality_score / 10,
            )
            for c in refs.body
        ],
    )


def synthesize(
    species: str,
    out_dir: Path,
    *,
    model_llm: str = "qwen2.5:14b-instruct",
    model_vlm: str = "qwen2.5vl:7b",
    lang: str = "fr",
) -> FactSheet:
    """
    Run end-to-end synthesis for a species, writing factsheet.json to out_dir.

    Loads papers.json and refs.json, generates a visual brief from body
    reference images via the VLM, sends the combined prompt to the LLM,
    injects visual_references, validates the result as a FactSheet, and
    persists factsheet.json.

    Args:
        species: The species name (e.g., "Tyrannosaurus rex").
        out_dir: Directory containing papers.json and optionally refs.json.
            factsheet.json will be written here.
        model_llm: Ollama model identifier for the text LLM.
        model_vlm: Ollama model identifier for the vision LLM.
        lang: Language code for text fields ("fr" or "en").

    Returns:
        The validated FactSheet instance.

    Raises:
        FileNotFoundError: If papers.json is missing from out_dir.
        dino_drawer.synthesis.ollama_client.SynthesisError: If the LLM fails
            to produce valid JSON after all retries.
        pydantic.ValidationError: If the LLM output does not satisfy the FactSheet schema.
    """
    out_dir = Path(out_dir)
    papers = PapersFile.model_validate_json((out_dir / "papers.json").read_text())
    refs = _load_refs(out_dir)

    visual_brief = ""
    if refs and refs.body:
        visual_brief = describe_body_refs(refs=refs, out_dir=out_dir, vlm_model=model_vlm)

    prompt = build_synthesis_prompt(
        species=species,
        papers=papers,
        refs=refs,
        visual_brief=visual_brief,
        lang=lang,
    )
    raw = call_llm_for_json(model=model_llm, prompt=prompt)

    visual_references = _build_visual_references(refs)
    raw["visual_references"] = visual_references.model_dump()

    _sanitize_source_ids(raw)

    fs = FactSheet.model_validate(raw)
    (out_dir / "factsheet.json").write_text(
        json.dumps(fs.model_dump(), indent=2, ensure_ascii=False)
    )
    return fs
