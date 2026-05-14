"""Build the synthesis prompt fed to the text LLM."""
from __future__ import annotations

import json

from dino_drawer.models import PapersFile, RefsFile

#: Maximum characters of the Wikipedia full article passed to the LLM. The
#: article can be 50-100 k chars; we send a chunk large enough to contain the
#: Description/Anatomy/Size sections (typically within the first ~10 k) without
#: blowing up the prompt token count.
_WIKI_CHAR_LIMIT = 12000


_SCHEMA_HINT = """\
You respond in strict JSON with EXACTLY these fields and types:

{
  "species": "string",
  "subtitle": "string",
  "dimensions": {
    "body_length": "string",
    "hip_height": "string",
    "skull_length": "string",
    "forelimb_length": "string",
    "tail_length": "string",
    "body_mass": "string",
    "source_ids": ["paper:N" | "wikipedia"]
  },
  "integument": {
    "integument_type": "string",
    "coloration": "string",
    "keratinous_structures": "string",
    "ontogenetic_variation": "string",
    "source_ids": ["paper:N" | "wikipedia"]
  },
  "posture": {
    "stance": "string",
    "typical_posture": "string",
    "locomotion_mode": "string",
    "source_ids": ["paper:N" | "wikipedia"]
  },
  "habitat": {
    "geological_period": "string",
    "biome": "string",
    "region_or_formation": "string",
    "source_ids": ["paper:N" | "wikipedia"]
  },
  "signature_traits": {
    "text": "string",
    "source_ids": ["paper:N" | "wikipedia"]
  },
  "conclusion": "string",
  "references": [{"id": "paper:0" | "wikipedia", "citation_short": "string", "doi": "string|null", "title": "string"}],
  "image_prompt": "string (English) — see strict constraints below"
}

General constraints:

- Every text field of the mandatory blocks (dimensions, integument, posture,
  habitat, signature_traits) must be **factual and sourced**. If a value is
  not findable in the provided corpus (Wikipedia + papers), write the literal
  string `"no data in corpus"` rather than inventing one. NEVER guess a
  numerical value.
- For numerical values (dimensions, mass), use **free-form text** with SI
  units and caveats: e.g. `"~12.3 m (specimen Sue); estimated 10-13 m for
  adults [paper:9][wikipedia]"`. Inline citations `[paper:N]` / `[wikipedia]`
  in the text body are ENCOURAGED.
- Each `source_ids` list contains the ids actually cited in the block text.
  Each id must exist in `references`.
- A block must contain at least 1 source_id, OR have all fields set to
  `"no data in corpus"` (in which case source_ids = []).

Constraints on `image_prompt` (English) — ONLY species-specific content.

A common universal prompt (wildlife photo style, head-on charge framing/pose,
theropod horizontal posture with counterweight tail, theropod skull anatomy
including scaly lips covering the teeth, camera settings, no-text footer) is
concatenated automatically at image-generation time. DO NOT REPEAT IT: no
directives about style, framing, pose, general posture, generic skull anatomy,
lips, camera, nor "no text".

Your `image_prompt` must describe EXCLUSIVELY:

1. **Paleoenvironment** (1 sentence): specific natural and geological setting,
   period, biome, time of day and light. E.g. `Late Cretaceous Laramidia
   conifer forest at dawn, mist drifting between trunks, sandy substrate with
   fallen needles`. Derived from the factsheet `habitat` block.

2. **Integument** by body mass (1-2 sentences, MANDATORY for non-avian
   theropods). Derived from the `integument` block:
   * Adults > 1 tonne (Tyrannosaurus, Allosaurus, Carnotaurus, etc.):
     `predominantly scaly skin across most of the body — dense pebbly small
     scales like a monitor lizard or crocodilian on flanks, belly, limbs and
     tail; a sparse mane-like ridge of filamentous protofeathers (hair-like
     quills, NEVER pennaceous flight feathers) runs along the dorsal midline
     from the nape to the upper tail base, visibly thin but present; NEVER a
     full downy coat; NEVER feathers covering the entire body`. Cf. Bell et
     al. 2017 (most skin is scaly) + Xu et al. 2012 (Yutyrannus, related
     tyrannosauroid with filaments).
   * < 100 kg or maniraptorans (Velociraptor, Microraptor, etc.): dense
     plumage, pennaceous feathers on arms and tail.
   * Intermediate sizes: mixed, justify from the papers.

3. **Color and pattern** (1 sentence): species-specific palette (markings,
   dorsal/ventral contrast, distinctive marks). Derived from
   `integument.coloration`.

4. **Unique anatomical features** (1-2 sentences): horns, sails, crests, arm
   length, unusual proportions. Derived from `signature_traits` + salient
   features in `dimensions` (e.g. T. rex tiny `forelimb_length`).

5. **Size** (short): approximate body length in meters, from
   `dimensions.body_length`.

6. **Predator state** (ONLY for active carnivorous theropods — NEVER for
   herbivores or omnivores): append a short sentence such as `a thin glistening
   trail of saliva at the corner of the slightly parted lips, faint dark blood
   smudges on the lower lip and chin (NOT on the teeth), as if the animal just
   made a kill seconds ago`. Adds the visual credibility of a super-predator
   mid-action while keeping the mouth nearly closed and the teeth hidden behind
   the lips.

Adapt if the species is NOT a non-avian theropod (bird, cetacean, mammal,
etc.): just describe the integument, coloration, and distinctive traits
without applying the rules above. Point 6 (saliva/blood) then applies only to
active mammalian/avian predators.

Length: 60-140 words.
"""


def build_synthesis_prompt(
    species: str,
    papers: PapersFile,
    refs: RefsFile | None,
    visual_brief: str,
    lang: str = "en",
) -> str:
    """
    Assemble the full synthesis prompt for the LLM.

    Args:
        species: The species name (e.g., "Tyrannosaurus rex").
        papers: PapersFile containing scientific papers and Wikipedia context.
        refs: Optional RefsFile with body visual references.
        visual_brief: Text describing visually-extracted features from reference images.
        lang: Language for text fields ("en" for English (default), "fr" for French).

    Returns:
        Complete synthesis prompt ready for the LLM.
    """
    paper_blocks = []
    for i, p in enumerate(papers.papers):
        paper_blocks.append(
            f"[paper:{i}] {', '.join(p.authors[:3]) or 'Anon'} ({p.year}). "
            f"{p.title}. DOI: {p.doi or 'n/a'}\nAbstract: {p.abstract[:600]}"
        )
    wiki_block = ""
    if papers.wikipedia:
        wiki_block = (
            f"[wikipedia] {papers.wikipedia.url}\n"
            f"{papers.wikipedia.extract[:_WIKI_CHAR_LIMIT]}"
        )

    refs_block = ""
    if refs and refs.body:
        refs_block = "Selected visual references: " + json.dumps(
            {"body": [r.path for r in refs.body]}
        )

    brief_block = (
        f"VISUAL_BRIEF (extracted from visual references):\n{visual_brief}"
        if visual_brief
        else ""
    )

    lang_directive = (
        "Write text fields in French."
        if lang == "fr"
        else "Write text fields in English."
    )

    return f"""\
You are a paleontology assistant. From the sources below, produce a structured information sheet for the species {species}.
{lang_directive}

SCIENTIFIC SOURCES:
{wiki_block}

{chr(10).join(paper_blocks) if paper_blocks else '(No recent papers.)'}

{refs_block}

{brief_block}

{_SCHEMA_HINT}
"""
