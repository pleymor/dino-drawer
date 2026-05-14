"""Test end-to-end synthesis with mocked LLM and VLM."""
import shutil
from pathlib import Path
from unittest.mock import patch

from dino_drawer.synthesis.main import synthesize

FIXTURES = Path(__file__).parent / "fixtures"


def test_synthesize_writes_valid_factsheet(tmp_path):
    """Synthesize from fixtures, mocking network calls, assert a valid factsheet is written."""
    shutil.copy(FIXTURES / "trex_papers.json", tmp_path / "papers.json")
    shutil.copy(FIXTURES / "trex_refs.json", tmp_path / "refs.json")
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "body_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    factsheet_dict = {
        "species": "Tyrannosaurus rex",
        "subtitle": "...",
        "dimensions": {
            "body_length": "12-13 m [paper:0]",
            "hip_height": "3.5-4 m [wikipedia]",
            "skull_length": "~1.5 m [paper:0]",
            "forelimb_length": "~1 m [wikipedia]",
            "tail_length": "~6 m [wikipedia]",
            "body_mass": "~8.8 t [paper:0]",
            "source_ids": ["paper:0", "wikipedia"],
        },
        "integument": {
            "integument_type": "predominantly scaly [paper:0]",
            "coloration": "dark dorsal, light ventral [wikipedia]",
            "keratinous_structures": "claws [wikipedia]",
            "ontogenetic_variation": "no data in corpus",
            "source_ids": ["paper:0", "wikipedia"],
        },
        "posture": {
            "stance": "bipedal [wikipedia]",
            "typical_posture": "horizontal spine, tail as counterweight [wikipedia]",
            "locomotion_mode": "cursorial [wikipedia]",
            "source_ids": ["wikipedia"],
        },
        "habitat": {
            "geological_period": "Late Cretaceous, ~68-66 Ma [wikipedia]",
            "biome": "coastal plain [wikipedia]",
            "region_or_formation": "Hell Creek Formation [wikipedia]",
            "source_ids": ["wikipedia"],
        },
        "signature_traits": {
            "text": "Massive boxy skull, tiny two-fingered forelimbs [wikipedia]",
            "source_ids": ["wikipedia"],
        },
        "conclusion": "...",
        "references": [
            {"id": "paper:0", "citation_short": "Cullen 2023", "doi": "10.1126/x", "title": "Lipped Tyrannosaurs"},
            {"id": "wikipedia", "citation_short": "Wikipedia", "doi": None, "title": "Tyrannosaurus"},
        ],
        "image_prompt": "photorealistic Tyrannosaurus rex, dark olive skin, sparse feathers on neck, dense forest, no text",
    }

    with patch("dino_drawer.synthesis.main.call_llm_for_json", return_value=factsheet_dict), \
         patch("dino_drawer.synthesis.main.describe_body_refs", return_value="Olive skin. Sparse neck feathers. Forest."):
        fs = synthesize(species="Tyrannosaurus rex", out_dir=tmp_path)

    assert fs.species == "Tyrannosaurus rex"
    assert (tmp_path / "factsheet.json").exists()
    assert len(fs.visual_references.body) == 1
    assert fs.visual_references.body[0].path == "refs/body_0.jpg"
    assert fs.dimensions.body_mass.startswith("~")
