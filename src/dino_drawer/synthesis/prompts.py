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
  "clade": "theropoda" | "sauropoda" | "stegosauria" | "ankylosauria" | "ceratopsia" | "ornithopoda" | "pachycephalosauria" | "other",
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

- The `clade` field is MANDATORY and selects which clade-specific block of
  the common hero prompt is used at image-gen time. Choose the closest
  morphological clade from the enum above:
    * `theropoda`         — bipedal predators with grasping hands (T. rex,
                            Allosaurus, Velociraptor, Carnotaurus, Spinosaurus,
                            Microraptor, etc.). DOES include feathered
                            maniraptorans (raptors, troodontids).
    * `sauropoda`         — gigantic quadrupedal long-neck herbivores
                            (Diplodocus, Brachiosaurus, Apatosaurus,
                            Argentinosaurus, Patagotitan, Nigersaurus, etc.).
    * `stegosauria`       — quadrupedal armoured herbivores with dorsal
                            plates and tail spikes (Stegosaurus, Kentrosaurus).
    * `ankylosauria`      — heavy quadrupedal "tank" herbivores with bony
                            armour plating, often a tail club (Ankylosaurus,
                            Euoplocephalus).
    * `ceratopsia`        — quadrupedal herbivores with frills and horns
                            (Triceratops, Styracosaurus, Protoceratops).
    * `ornithopoda`       — facultative bipedal herbivores with beak / duck
                            bill (Iguanodon, Parasaurolophus, Edmontosaurus,
                            Hypsilophodon).
    * `pachycephalosauria` — bipedal dome-headed herbivores
                            (Pachycephalosaurus).
    * `other`             — fallback for anything that does not cleanly fit
                            one of the above (very basal forms, ambiguous, or
                            non-dinosaur species).
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

A common universal prompt is concatenated automatically at image-generation
time. It contains: wildlife photo style, framing, camera settings, no-text
footer, AND a clade-specific block (posture, anatomy, lips/beak, predator
or browser state) chosen from the `clade` field. DO NOT REPEAT this block:
no directives about photo style, framing, camera, generic anatomy by clade,
lips/beak, or "no text".

Your `image_prompt` must describe EXCLUSIVELY:

1. **Paleoenvironment** (1 sentence): specific natural and geological setting,
   period, biome, time of day and light. E.g. `Late Cretaceous Laramidia
   conifer forest at dawn, mist drifting between trunks, sandy substrate with
   fallen needles`. Derived from the `habitat` block.

2. **Integument** (1-2 sentences). Derived from the `integument` block:
   * **Large theropods** (> 1 t, `clade=theropoda` and adult mass big):
     `predominantly scaly skin across most of the body — dense pebbly small
     scales like a monitor lizard on flanks, belly, limbs and tail; sparse
     mane-like ridge of filamentous protofeathers (hair-like quills, never
     pennaceous) along the dorsal midline from the nape to the upper tail
     base`. Cf. Bell et al. 2017 + Xu et al. 2012.
   * **Small theropods / maniraptorans** (< 100 kg): dense plumage,
     pennaceous feathers on arms and tail.
   * **Sauropods (clade=sauropoda)**: thick scaly hide, smooth back by
     default. DO NOT claim dorsal spines, scutes, ridges, or osteoderms
     UNLESS the scientific sources (papers/Wikipedia) explicitly document
     them for that species. Many artistic reconstructions show small spines
     but most lack fossil evidence — write `keratinous_structures` as
     `"no data in corpus"` if no documented evidence; do NOT invent them
     from visual_brief alone (artists' interpretations are not data).
     Documented exceptions include: some titanosaurs (small isolated
     osteoderms), Amargasaurus (bifid neural spines on neck).
   * **Stegosaurs, ankylosaurs, ceratopsians, ornithopods,
     pachycephalosaurs**: thick scaly hide with osteoderms/keratinous
     structures (plates, spikes, frill, dome, beak) as appropriate per
     the species; NEVER feathers (these clades have no evidence of plumage).

3. **Color and pattern** (1 sentence): species-specific palette. Derived from
   `integument.coloration`.

4. **Unique anatomical features** (1-2 sentences): horns, sails, crests,
   plates, spikes, tail club, frill, dome, beak shape, arm length, unusual
   proportions. Derived from `signature_traits` + salient `dimensions`.

5. **Size** (short): approximate body length in meters, from
   `dimensions.body_length`.

6. **State cue** (1 short sentence):
   * For active carnivorous theropods only: `a thin glistening trail of
     saliva at the corner of the slightly parted lips, faint dark blood
     smudges on the lower lip and chin (NOT on the teeth), as if the animal
     just made a kill seconds ago`.
   * For herbivores (sauropods, stegosaurs, ankylosaurs, ceratopsians,
     ornithopods, pachycephalosaurs): describe a calm browsing state, e.g.
     `head lowered to crop foliage` or `head raised, chewing on a branch`,
     with no aggressive cues.

Adapt if the species is NOT a non-avian dinosaur (bird, cetacean, mammal,
etc.): describe integument/coloration/distinctive traits freely; the clade
should be `other` so the image-gen prompt skips the dinosaur-specific anatomy
block.

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
