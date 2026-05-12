"""Build the synthesis prompt fed to the text LLM."""
from __future__ import annotations

import json

from dino_drawer.models import PapersFile, RefsFile


_SCHEMA_HINT = """\
Tu réponds en JSON strict avec EXACTEMENT ces champs et types :

{
  "species": "string",
  "subtitle": "string",
  "annotations": [
    {"region": "tête" | "peau_et_couverture" | "membres_anterieurs" | "membres_posterieurs" | "queue",
     "facts": ["string", ...],
     "source_ids": ["paper:N" | "wikipedia"]}
  ],
  "skull_view": {"facts": ["string"], "scale_cm": number, "source_ids": ["paper:N" | "wikipedia"]},
  "size": {"length_m": [min, max], "hip_height_m": [min, max], "source_ids": ["paper:N" | "wikipedia"]},
  "conclusion": "string",
  "references": [{"id": "paper:0" | "wikipedia", "citation_short": "string", "doi": "string|null", "title": "string"}],
  "image_prompt": "string en anglais, photoréaliste, mentionnant >= 3 éléments visuels observés, finit par 'no text'"
}

Contraintes :
- Inclure TOUTES les régions anatomiques ci-dessus. Si tu n'as pas d'info, écris un fait baseline et cite "wikipedia".
- Chaque source_id doit exister dans `references`.
- `image_prompt` doit être en anglais et doit mentionner au moins 3 éléments visuels du visual_brief.
"""


def build_synthesis_prompt(
    species: str,
    papers: PapersFile,
    refs: RefsFile | None,
    visual_brief: str,
    lang: str = "fr",
) -> str:
    """
    Assemble the full synthesis prompt for the LLM.

    Args:
        species: The species name (e.g., "Tyrannosaurus rex").
        papers: PapersFile containing scientific papers and Wikipedia context.
        refs: Optional RefsFile with visual references.
        visual_brief: Text describing visually-extracted features from reference images.
        lang: Language for text fields ("fr" for French, other for English).

    Returns:
        Complete synthesis prompt ready for the LLM.
    """
    paper_blocks = []
    for i, p in enumerate(papers.papers):
        paper_blocks.append(
            f"[paper:{i}] {', '.join(p.authors[:3]) or 'Anon'} ({p.year}). "
            f"{p.title}. DOI: {p.doi or 'n/a'}\nRésumé : {p.abstract[:600]}"
        )
    wiki_block = ""
    if papers.wikipedia:
        wiki_block = f"[wikipedia] {papers.wikipedia.url}\n{papers.wikipedia.extract[:1500]}"

    refs_block = ""
    if refs and (refs.body or refs.skull):
        refs_block = "Références visuelles retenues : " + json.dumps(
            {"body": [r.path for r in refs.body], "skull": [r.path for r in refs.skull]}
        )

    brief_block = f"VISUAL_BRIEF (extrait des refs visuelles) :\n{visual_brief}" if visual_brief else ""

    lang_directive = "Rédige les champs textuels en français." if lang == "fr" else "Write text fields in English."

    return f"""\
Tu es un assistant paléontologique. À partir des sources ci-dessous, produis une fiche d'information structurée sur l'espèce {species}.
{lang_directive}

SOURCES SCIENTIFIQUES :
{wiki_block}

{chr(10).join(paper_blocks) if paper_blocks else '(Aucun papier récent.)'}

{refs_block}

{brief_block}

{_SCHEMA_HINT}
"""
