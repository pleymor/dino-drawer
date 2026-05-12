"""Test end-to-end synthesis with mocked LLM and VLM."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch

from dino_drawer.synthesis.main import synthesize

FIXTURES = Path(__file__).parent / "fixtures"


def test_synthesize_writes_valid_factsheet(tmp_path):
    """Synthesize from fixtures, mocking network calls, and assert a valid factsheet is written."""
    shutil.copy(FIXTURES / "trex_papers.json", tmp_path / "papers.json")
    shutil.copy(FIXTURES / "trex_refs.json", tmp_path / "refs.json")
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "body_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    factsheet_dict = {
        "species": "Tyrannosaurus rex",
        "subtitle": "...",
        "annotations": [
            {"region": "tête", "facts": ["Lips covering teeth"], "source_ids": ["paper:0"]},
            {"region": "peau_et_couverture", "facts": ["Scaly skin"], "source_ids": ["wikipedia"]},
            {"region": "membres_anterieurs", "facts": ["Short"], "source_ids": ["wikipedia"]},
            {"region": "membres_posterieurs", "facts": ["Strong"], "source_ids": ["wikipedia"]},
            {"region": "queue", "facts": ["Long"], "source_ids": ["wikipedia"]},
        ],
        "skull_view": {"facts": ["large"], "scale_cm": 50, "source_ids": ["wikipedia"]},
        "size": {"length_m": [12, 13], "hip_height_m": [3.5, 4], "source_ids": ["wikipedia"]},
        "conclusion": "...",
        "references": [
            {"id": "paper:0", "citation_short": "Cullen 2023", "doi": "10.1126/x", "title": "Lipped Tyrannosaurs"},
            {"id": "wikipedia", "citation_short": "Wikipedia", "doi": None, "title": "Tyrannosaurus"},
        ],
        "image_prompt": "photorealistic Tyrannosaurus rex, dark olive skin, sparse feathers on neck, dense forest, no text",
    }

    with patch("dino_drawer.synthesis.main.call_llm_for_json", return_value=factsheet_dict), \
         patch("dino_drawer.synthesis.main.describe_body_refs", return_value="Peau olive. Plumes éparses cou. Forêt."):
        fs = synthesize(species="Tyrannosaurus rex", out_dir=tmp_path)

    assert fs.species == "Tyrannosaurus rex"
    assert (tmp_path / "factsheet.json").exists()
    assert len(fs.visual_references.body) == 1
    assert fs.visual_references.body[0].path == "refs/body_0.jpg"
