"""Tests for the DinoDrawerAgent orchestration (all sub-steps mocked)."""
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from dino_drawer.agent import DinoDrawerAgent
from dino_drawer.models import (
    PapersFile, RawImagesFile, RefsFile, FactSheet, Reference, VisualReferences,
    Dimensions, Integument, Posture, Habitat, SignatureTraits,
)


def _papers() -> PapersFile:
    return PapersFile(species="X", wikipedia=None, papers=[])


def _refs_raw() -> RawImagesFile:
    return RawImagesFile(species="X", images=[])


def _refs() -> RefsFile:
    return RefsFile(species="X", body=[])


def _factsheet() -> FactSheet:
    return FactSheet(
        species="X", subtitle="",
        dimensions=Dimensions(
            body_length="x", hip_height="x", skull_length="x",
            forelimb_length="x", tail_length="x", body_mass="x",
            source_ids=["wikipedia"],
        ),
        integument=Integument(
            integument_type="x", coloration="x",
            keratinous_structures="x", ontogenetic_variation="x",
            source_ids=["wikipedia"],
        ),
        posture=Posture(
            stance="x", typical_posture="x", locomotion_mode="x",
            source_ids=["wikipedia"],
        ),
        habitat=Habitat(
            geological_period="x", biome="x", region_or_formation="x",
            source_ids=["wikipedia"],
        ),
        signature_traits=SignatureTraits(text="x", source_ids=["wikipedia"]),
        conclusion="",
        references=[Reference(id="wikipedia", citation_short="W", doi=None, title="t")],
        visual_references=VisualReferences(body=[]),
        image_prompt="..",
    )


@pytest.mark.asyncio
async def test_agent_runs_full_pipeline(tmp_path):
    with patch("dino_drawer.agent.papers_aggregator", AsyncMock(return_value=_papers())) as mp, \
         patch("dino_drawer.agent.images_aggregator", AsyncMock(return_value=_refs_raw())) as mi, \
         patch("dino_drawer.agent.classify_and_select", return_value=_refs()) as mc, \
         patch("dino_drawer.agent.synthesize", return_value=_factsheet()) as ms, \
         patch("dino_drawer.agent.generate_assets") as mg:
        agent = DinoDrawerAgent(out_root=tmp_path)
        out = await agent.run("Tyrannosaurus rex")
    assert out.name == "hero.png"
    mp.assert_awaited_once()
    mi.assert_awaited_once()
    mc.assert_called_once()
    ms.assert_called_once()
    mg.assert_called_once()


@pytest.mark.asyncio
async def test_agent_skips_existing_steps(tmp_path):
    species_dir = tmp_path / "tyrannosaurus-rex"
    species_dir.mkdir()
    (species_dir / "papers.json").write_text(_papers().model_dump_json())

    with patch("dino_drawer.agent.papers_aggregator", AsyncMock(return_value=_papers())) as mp, \
         patch("dino_drawer.agent.images_aggregator", AsyncMock(return_value=_refs_raw())) as mi, \
         patch("dino_drawer.agent.classify_and_select", return_value=_refs()), \
         patch("dino_drawer.agent.synthesize", return_value=_factsheet()), \
         patch("dino_drawer.agent.generate_assets"):
        agent = DinoDrawerAgent(out_root=tmp_path)
        await agent.run("Tyrannosaurus rex")
    mp.assert_not_awaited()
    mi.assert_awaited_once()
