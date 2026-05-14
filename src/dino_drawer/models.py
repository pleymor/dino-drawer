"""Pydantic models for every pipeline artifact (papers, refs, factsheet)."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# --- Step 1: papers ---

class Paper(BaseModel):
    """A scientific publication fetched from one of the research sources."""
    doi: str | None
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    source: str  # "semantic_scholar" | "openalex" | "wikipedia"


class WikipediaContext(BaseModel):
    """Wikipedia article context used as baseline knowledge."""
    url: str
    extract: str


class PapersFile(BaseModel):
    """Collection of papers and Wikipedia context for a species."""
    species: str
    wikipedia: WikipediaContext | None
    papers: list[Paper]


# --- Step 2: raw images ---

class RawImage(BaseModel):
    """An image candidate downloaded but not yet filtered."""
    id: int
    path: str  # relative to species output dir
    source_url: str
    source: str  # "wikimedia_commons" | "wikipedia" | "phylopic" | "inaturalist"
    credit: str
    license: str
    width: int
    height: int
    search_query: str


class RawImagesFile(BaseModel):
    """Collection of raw unfiltered images for a species."""
    species: str
    images: list[RawImage]


# --- Step 3: classified images ---

class ClassifiedImage(BaseModel):
    """A raw image after VLM classification."""
    id: int
    path: str  # in refs/ if kept
    type: str
    view: str
    usable_for_body_generation: bool
    realism_score: float
    quality_score: float
    description_courte: str
    credit: str
    license: str


class RefsFile(BaseModel):
    """Collection of classified body reference images for a species."""
    species: str
    body: list[ClassifiedImage] = Field(default_factory=list)
    rejected_count: int = 0


# --- Step 4: factsheet ---
#
# The factsheet is structured around mandatory "must-have" blocks that drive
# the hero illustration. Every block has its own ``source_ids`` list so the
# validator can verify each block is grounded in the corpus (or explicitly
# marked "no data in corpus" inside the text).

class Dimensions(BaseModel):
    """Physical dimensions of the species. Free-text fields with caveats/ranges."""
    body_length: str        # snout-to-tail
    hip_height: str
    skull_length: str
    forelimb_length: str    # distinctive for some clades (T. rex tiny arms)
    tail_length: str
    body_mass: str
    source_ids: list[str]


class Integument(BaseModel):
    """Skin / feathers / coloration / keratinous structures."""
    integument_type: str            # scaly | feathered | mixed (+ distribution)
    coloration: str                 # dorsal/ventral palette + patterns
    keratinous_structures: str      # horns, crests, beaks, claws, plates
    ontogenetic_variation: str      # juvenile vs adult differences
    source_ids: list[str]


class Posture(BaseModel):
    """Stance, posture and locomotion mode."""
    stance: str             # bipedal | quadrupedal | facultative
    typical_posture: str    # horizontal spine, upright, sprawling…
    locomotion_mode: str    # cursorial | ambulatory | arboreal | aquatic
    source_ids: list[str]


class Habitat(BaseModel):
    """Geological context and biome — drives the image background."""
    geological_period: str          # "Late Cretaceous (Maastrichtian), 68-66 Ma"
    biome: str                      # forest | plain | coast | swamp | desert
    region_or_formation: str        # "Hell Creek Formation, Laramidia"
    source_ids: list[str]


class SignatureTraits(BaseModel):
    """Free-form description of the visually distinctive features of the species."""
    text: str
    source_ids: list[str]


class Reference(BaseModel):
    """A bibliographic reference cited in the factsheet."""
    id: str
    citation_short: str
    doi: str | None
    title: str


class VisualRef(BaseModel):
    """A visual reference image with metadata."""
    path: str
    credit: str
    license: str
    score: float


class VisualReferences(BaseModel):
    """Body reference images used to condition the hero generation."""
    body: list[VisualRef] = Field(default_factory=list)


class FactSheet(BaseModel):
    """Complete factsheet for a species with mandatory blocks and references."""
    species: str
    subtitle: str
    dimensions: Dimensions
    integument: Integument
    posture: Posture
    habitat: Habitat
    signature_traits: SignatureTraits
    conclusion: str
    references: list[Reference]
    visual_references: VisualReferences
    image_prompt: str

    @model_validator(mode="after")
    def _every_block_must_have_known_sources(self) -> FactSheet:
        """Every block's source_ids must match a Reference id."""
        known = {r.id for r in self.references}
        blocks: list[tuple[str, list[str]]] = [
            ("dimensions", self.dimensions.source_ids),
            ("integument", self.integument.source_ids),
            ("posture", self.posture.source_ids),
            ("habitat", self.habitat.source_ids),
            ("signature_traits", self.signature_traits.source_ids),
        ]
        for name, sids in blocks:
            for sid in sids:
                if sid not in known:
                    raise ValueError(f"Block '{name}' references unknown source {sid!r}")
        return self
