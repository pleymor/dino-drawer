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
  "image_prompt": "string en anglais — voir contraintes strictes ci-dessous"
}

Contraintes générales :
- Inclure TOUTES les régions anatomiques ci-dessus. Si tu n'as pas d'info, écris un fait baseline et cite "wikipedia".
- Chaque source_id doit exister dans `references`.

Contraintes STRICTES sur `image_prompt` (en anglais) :
- Style : `unedited DSLR wildlife photograph`, `RAW image`, `national geographic style`, `hyper-realistic skin pores and feather texture`, `subtle ISO grain`, `natural lens distortion`, `slightly imperfect framing as if shot in the field`. JAMAIS `illustration`, `drawing`, `painting`, `paleoart`, `artwork`, `render`, `CGI`, `3D model`, `studio shot`.
- Framing OBLIGATOIRE : `front-facing perspective, the animal is charging or lunging straight toward the camera, head pointed at the viewer, three-quarter front view or near head-on, the entire body visible — head, body, legs and most of the tail in frame, no cropping of the head or shoulders, animal occupies 60-80% of the frame`.
- Pose AGGRESSIVE : `aggressive charge toward the viewer, head thrust forward, muscles bunched, claws splayed, body weight shifted to the front leg, low menacing glare directly at the camera, neck feathers raised`. La sensation doit être celle d'un témoin qui prend la photo une seconde avant de se faire attaquer.
- Environnement OBLIGATOIRE : décor naturel spécifique au paléoenvironnement (forêt, plaine, marais, etc. avec contexte géologique : ex. "Late Cretaceous Laramidia conifer forest at dawn"). JAMAIS de fond uni / fond blanc / studio.
- Bouche et lèvres OBLIGATOIRES (si l'espèce a des lèvres scaleuses comme les théropodes non-aviens d'après les papiers récents) : `mouth firmly shut, lip line forming a continuous scaly seam from snout tip to jaw corner, like a closed-mouth monitor lizard or a resting Komodo dragon, ABSOLUTELY ZERO teeth visible, not even a hint of a tooth tip, no fangs poking out, the upper lip overhangs and conceals the entire dental row from the outside`. Si l'espèce a une morphologie buccale différente (oiseau, cétacé, mammifère), adapte. Ne JAMAIS écrire "baring teeth" ni "snarling" ni "mouth agape".
- Détails anatomiques OBLIGATOIRES : reprend littéralement les faits du factsheet (répartition des plumes/écailles, couleur de peau, longueur de queue, posture, etc.). Cite-en au moins 6.
- Camera : `eye-level`, `35mm prime lens`, `shallow depth of field`, `golden hour` ou `overcast soft light`.
- Termine TOUJOURS par : `absolutely no text, no captions, no watermark, no signature, no UI overlays, no logo`.
- Longueur : 80-150 mots.
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
