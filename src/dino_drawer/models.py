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
    usable_for_skull_generation: bool
    realism_score: float
    quality_score: float
    description_courte: str
    credit: str
    license: str


class RefsFile(BaseModel):
    """Collection of classified reference images for a species."""
    species: str
    body: list[ClassifiedImage] = Field(default_factory=list)
    skull: list[ClassifiedImage] = Field(default_factory=list)
    rejected_count: int = 0


# --- Step 4: factsheet ---

class Annotation(BaseModel):
    """An annotated anatomical region with facts sourced from references."""
    region: str  # tête | peau_et_couverture | membres_anterieurs | membres_posterieurs | queue
    facts: list[str]
    source_ids: list[str]


class SkullView(BaseModel):
    """Details about the skull with facts sourced from references."""
    facts: list[str]
    scale_cm: float
    source_ids: list[str]


class Size(BaseModel):
    """Size measurements (length and hip height) sourced from references."""
    length_m: list[float]  # [min, max]
    hip_height_m: list[float]  # [min, max]
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
    """Collection of visual references for body and skull."""
    body: list[VisualRef] = Field(default_factory=list)
    skull: list[VisualRef] = Field(default_factory=list)


class FactSheet(BaseModel):
    """Complete factsheet for a species with annotations, references, and visual aids."""
    species: str
    subtitle: str
    annotations: list[Annotation]
    skull_view: SkullView
    size: Size
    conclusion: str
    references: list[Reference]
    visual_references: VisualReferences
    image_prompt: str

    @model_validator(mode="after")
    def _every_fact_must_have_a_known_source(self) -> FactSheet:
        """Validate that every fact references an existing source by id."""
        known = {r.id for r in self.references}
        for ann in self.annotations:
            for sid in ann.source_ids:
                if sid not in known:
                    raise ValueError(f"Annotation '{ann.region}' references unknown source {sid!r}")
        for sid in self.skull_view.source_ids + self.size.source_ids:
            if sid not in known:
                raise ValueError(f"Unknown source id {sid!r}")
        return self
