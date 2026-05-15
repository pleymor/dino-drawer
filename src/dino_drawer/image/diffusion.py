"""Gemini-backed hero image generation. Produces hero.png + image_meta.json.

The hero prompt is composed at gen time from three layers:

  [_UNIVERSAL_PREAMBLE] + [_CLADE_PROMPTS[factsheet.clade]] + [species-specific] + [_NO_TEXT_FOOTER]

* The universal preamble is the wildlife-photo aesthetic + camera settings —
  same for every species.
* The clade block carries anatomy, posture and state directives appropriate
  to the broad morphological group (theropoda, sauropoda, stegosauria,
  ankylosauria, ceratopsia, ornithopoda, pachycephalosauria, or a minimal
  ``other`` fallback). Picked from ``factsheet.clade``.
* The species-specific block is the LLM-generated ``factsheet.image_prompt``.
* The footer kills text/captions/watermarks.

Adding a new clade = adding one entry to ``_CLADE_PROMPTS``.

Alongside the image, we write ``image_meta.json`` recording which image model
was used and when. ``publish`` reads this so the catalog tracks which model
produced each species' image.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from dino_drawer.clients.gemini import GeminiClient, _image_model
from dino_drawer.models import FactSheet


def _load_ref_paths(out_dir: Path, paths: list[str]) -> list[Path]:
    """Resolve relative ref paths against out_dir; skip any missing files."""
    out: list[Path] = []
    for p in paths:
        full = out_dir / p
        if full.exists():
            out.append(full)
    return out


# ---------------------------------------------------------------------------
# Universal preamble — applies to every species regardless of clade.
# ---------------------------------------------------------------------------

_UNIVERSAL_PREAMBLE = (
    # Medium & style.
    "Award-winning unedited wildlife photograph, RAW DSLR image, National Geographic "
    "feature-quality, shot on full-frame 200mm telephoto prime lens (classic wildlife "
    "focal length: compressed background, subject filling much of the frame, towering "
    "presence), subtle ISO grain, natural lens distortion, framing slightly imperfect "
    "as if captured by a witness in the field. ABSOLUTELY NOT a wide-angle shot, NOT a "
    "fisheye, NOT a close-up macro. ABSOLUTELY NOT an illustration, drawing, painting, "
    "render, CGI, 3D model, paleoart, or studio shot. "
    # Framing — body-size-aware. Small/mid dinos: full body. Giants: drama wins over full-body.
    "Framing depends on the animal's size:\n"
    "  - Small to mid-sized animals (under ~6 m body length, under ~3 m hip height): the "
    "ENTIRE body must be in frame — head, body, legs and the full tail visible from base "
    "to tip; no cropping of the head or shoulders; animal occupies 60-80% of the frame.\n"
    "  - Large animals (>6 m body length, e.g. large theropods, ceratopsians, ankylosaurs): "
    "still favour full-body, but slight cropping of the tail tip is acceptable if it "
    "improves the dramatic angle.\n"
    "  - GIANTS (>15 m body length or >5 m hip height — sauropods, very large theropods): "
    "the photographer is STANDING CLOSE to the animal (5-12 m away), so it is impossible "
    "to fit everything in frame. PRIORITY: head, neck, much of the body, and at least the "
    "FRONT LEGS must be clearly visible. The whip-end of the tail can EXTEND OUT of the "
    "frame on one side, the back foot likewise. The dramatic close-encounter angle wins "
    "over completeness — the viewer must feel like they are standing right next to a "
    "creature that is much bigger than them. "
    # Scale reference via vegetation/terrain.
    "The scene includes MATURE adult-sized vegetation in the same frame as a natural "
    "scale reference — tall conifers (20-40 m), large fern trees, mature cycads, or "
    "comparable period-appropriate flora. The eye must be able to gauge the animal's "
    "size from the surrounding flora and terrain features (boulders, fallen trunks, "
    "watercourses). Do NOT use only low scrubby bushes or tiny ferns that give no "
    "sense of scale. "
    # Camera height & angle — anchored to adult human eye-level, tilted UP for big dinos.
    "Camera held at adult human standing eye-height — approximately 1.6 to 1.8 m above "
    "the ground, NEVER higher (NOT a bird's-eye or aerial view). The camera angle is "
    "TILTED according to the animal's size: small dinosaurs (under ~1 m) seen from "
    "slightly above as we look down; mid-sized (~2-3 m) at level gaze; large theropods "
    "(4-6 m hip) photographed in clear LOW-ANGLE (contre-plongée) with the camera "
    "tilted ~20-30° upward — the animal looms over the viewer, we see the underside "
    "of the jaw and chest from below. For GIANTS (>5 m hip — sauropods, very large "
    "theropods), the contre-plongée is EXTREME: the camera tilts 35-50° upward; the "
    "belly, ribcage underside and underside of the long neck are clearly visible from "
    "below; the head and neck soar high into the frame against sky or canopy; the "
    "back/topline is NOT seen (we are physically below it). The horizon sits in the "
    "lower third of the frame. The viewer feels physically small next to a creature "
    "that towers above. "
    # Depth of field — DEEP focus, everything in the scene sharp. Stopped-down aperture.
    "Depth of field: aperture stopped down to f/11–f/16 for DEEP focus across the "
    "entire scene. Subject AND background are both in sharp focus — every leaf, every "
    "trunk, every distant tree, every fern, every rock is crisply rendered with "
    "visible detail. NO part of the frame is blurred. ABSOLUTELY NO bokeh, NO bokeh "
    "balls, NO cinematic shallow-DOF, NO defocused background, NO mist haze softening "
    "distant trees. Telephoto compression makes the background feel close but it is "
    "NOT softened. Think landscape-photography sharpness throughout: foreground "
    "vegetation crisp, the animal crisp, background trees and sky crisp. "
    # Lighting & atmosphere — strong directional light, high dynamic range.
    "Lighting is STRONG and directional — low-angle golden-hour sunlight "
    "with warm rim light catching the silhouette, crisp side shadows, deep blacks in "
    "the shaded areas, and vivid saturated highlights. High dynamic range. Subject skin "
    "texture is sharply resolved with microcontrast. "
    "AVOID flat washed-out lighting; AVOID overcast hazy mood; AVOID muted desaturated "
    "palette; AVOID a dull lifeless atmosphere. "
)

_NO_TEXT_FOOTER = (
    "ABSOLUTELY NO TEXT anywhere — no captions, no watermark, no signature, "
    "no UI overlays, no logo, no labels."
)


# ---------------------------------------------------------------------------
# Clade-specific prompt blocks — anatomy, posture and state per morphology.
# ---------------------------------------------------------------------------

_CLADE_THEROPODA = (
    # Pose & framing
    "The animal is moving briskly toward the camera in a three-quarter front view, "
    "head thrust forward and pointed at the viewer, muscles bunched, claws splayed, "
    "body weight on the front leg, focused predatory stare locked on the camera. "
    "Mouth FULLY CLOSED, jaws firmly shut — NOT parted, NOT a snarl, NOT a roar, "
    "NOT a pant; the lips meet along their entire length forming a clean continuous "
    "scaly seam from snout tip to jaw corner. "
    # Posture
    "Body held horizontally, spine roughly parallel to the ground, "
    "massive head and long muscular tail balance each other around the hips like a seesaw, "
    "tail extended horizontally behind the body and clearly visible from base to tip, "
    "the head-to-tail line is essentially level. "
    # Skull
    "Skull anatomy based on modern CT scans of the most complete specimens: "
    "forward-facing eyes producing strong binocular overlap, external nares on the rostrum "
    "near the snout tip, deep boxy skull profile. "
    # Lips (Cullen et al. 2023) — closed-mouth rule: ZERO teeth visible.
    "Theropod lips: the skull is sheathed in scaly extra-oral tissue (lips) that FULLY "
    "covers the dental row, like a modern monitor lizard or Komodo dragon at rest. "
    "Because the jaws are shut, the lips are in contact along their ENTIRE length and "
    "ABSOLUTELY ZERO TEETH ARE VISIBLE anywhere on the face. No tooth tips, no fangs, "
    "no premaxillary points peeking out, no enamel glint, no white spot on the mouth "
    "line, no tooth showing through a gap in the lips. The mouth must read visually as "
    "a clean unbroken scaly seam — indistinguishable in silhouette from a closed-mouth "
    "monitor lizard. NEVER show side teeth; NEVER show front teeth; NEVER show ANY "
    "tooth root-to-tip; NEVER a Jurassic-Park grin. "
    # Dental anatomy (concealed but anatomically present)
    "Dental anatomy (hidden under the lips but anatomically correct underneath): upper "
    "teeth are SHORT relative to snout depth and DO NOT extend below the lower jaw lip "
    "line. Both jaws carry a COMPLETE row of teeth — never gaps, never missing front "
    "teeth, never a toothless snout tip — but the entire dental row stays fully "
    "concealed behind the lip seal in this view. "
)

_CLADE_SAUROPODA = (
    # Pose & framing — HIGH BROWSING is the default for most large sauropods.
    "The animal stands in three-quarter side view, calmly browsing — not aggressive. "
    "Pillar-like front and hind limbs support the massive trunk. The long whip-like "
    "tail extends horizontally behind the body. The trees in the frame must be "
    "ADULT-SIZED, tall enough to dwarf the dinosaur and give an instant scale reference. "
    # Sub-family silhouette — critical, brachiosaurids and diplodocids look DIFFERENT.
    "Silhouette must match the species' family:\n"
    "  - Brachiosauridae (Brachiosaurus, Giraffatitan, Abydosaurus, Lusotitan): "
    "FRONT LIMBS ARE NOTABLY LONGER THAN HIND LIMBS — the withers sit HIGH, the hips "
    "sit LOW, the back SLOPES DOWN from shoulders to hips (giraffe-like). The neck is "
    "held NEARLY VERTICAL, reaching straight up into the high canopy (head 12-15 m up). "
    "Body more slender than rotund, NOT barrel-shaped.\n"
    "  - Diplodocidae (Diplodocus, Apatosaurus, Brontosaurus, Barosaurus, Supersaurus, "
    "Amphicoelias) and Dicraeosauridae: FRONT AND HIND LIMBS ARE ROUGHLY EQUAL "
    "LENGTH; back is HORIZONTAL or slightly arched over the hips. Neck held nearly "
    "HORIZONTAL, parallel to the ground — feeding at low-to-mid height. Body long, "
    "deep, more cylindrical.\n"
    "  - Rebbachisauridae (Nigersaurus): low ground-grazer, neck held low and "
    "horizontal, wide square-tipped snout cropping at ground level.\n"
    "  - Titanosauria (Argentinosaurus, Patagotitan, Nagatitan): typically "
    "robust, broad chest, columnar limbs. Neck length VARIES strongly inside "
    "the clade — most titanosaurs have moderate necks, but Lognkosaurians "
    "(Patagotitan, Argentinosaurus) have EXCEPTIONALLY LONG necks arched in an "
    "elegant S-curve, head ~12-15 m above ground, comparable in length to a "
    "brachiosaurid's neck. Follow the species block's explicit description.\n"
    "Adapt the high-browse pose to match the family above. Brachiosaurids reach "
    "STRAIGHT UP; diplodocids/dicraeosaurids reach FORWARD or down. "
    # Skull & head
    "Skull is proportionally tiny relative to the body, mounted on the end of the long "
    "neck. Eyes laterally placed (not forward-binocular). Mouth holds peg-like or "
    "spatulate teeth; no fangs. Mouth closed or gently chewing leaves — NEVER snarling. "
    # Integument — IMPORTANT: smooth hide is the default, no spines.
    "Thick scaly hide across the entire body, similar to a modern elephant or "
    "crocodile. NO feathers, NO plumage, NO dorsal mane. By DEFAULT the back is SMOOTH "
    "— NO dorsal spines, NO triangular scutes, NO continuous ridge of osteoderms along "
    "the back. ONLY add dorsal armature if the species-specific block EXPLICITLY "
    "documents it from fossil evidence (e.g. some titanosaurs had isolated osteoderms; "
    "Amargasaurus had bifid neural spines). "
)

_CLADE_STEGOSAURIA = (
    # Pose & framing — strong anti-paleoart bias, dynamic stance
    "The animal stands in a DYNAMIC three-quarter side view (NEVER a flat lateral "
    "profile from a textbook), body angled toward the camera at ~30-45°, head turned "
    "slightly to look around or browse. Quadrupedal stance, one foot subtly forward "
    "as if mid-step, head held low. Calm browsing posture, NOT aggressive. "
    "CRITICAL: this is a wildlife PHOTOGRAPH (think National Geographic in-situ "
    "footage of a real animal), NOT a museum illustration, NOT a paleoart side "
    "profile, NOT a children's book diagram. The body must show foreshortening; "
    "skin texture, plate edges and tail spikes must show real-world surface detail "
    "(grime, asymmetric wear, slight asymmetric shadow). "
    # Posture & body plan
    "Quadrupedal, hindlimbs noticeably taller than the front limbs so the back arches "
    "highest over the hips. Body is wide and barrel-shaped, low to the ground at the "
    "shoulders. Tail held off the ground, level or slightly raised. "
    # Plates and spikes — defining trait
    "Dorsal armature: a DOUBLE alternating row of bony plates and/or tall spikes runs "
    "along the spine from the neck to the tail. The plates are the species' iconic "
    "feature — render them prominently. The tail terminates in two pairs of long sharp "
    "spikes (the 'thagomizer'), pointed slightly sideways and back. "
    # Skull & head
    "Skull is small and elongated relative to the body, with a narrow tapered snout "
    "ending in a horny BEAK (rhamphotheca) — NO long fangs visible, NO predator teeth. "
    "Mouth closed or chewing leaves. Eyes laterally placed. "
    # Integument
    "Thick scaly herbivore hide across the body; no feathers, no plumage. "
)

_CLADE_ANKYLOSAURIA = (
    # Pose & framing
    "The animal stands in three-quarter side view, low and wide, head close to the "
    "ground browsing on low vegetation. Defensive but calm posture, body angled "
    "slightly toward the camera. "
    # Posture & body plan
    "Quadrupedal, very low and broad — width almost equals height. Short stocky limbs. "
    "Tail extends behind, terminating in a heavy BONY CLUB (in advanced ankylosaurids) "
    "or stiff with paired spikes (in nodosaurids). "
    # Armour — defining trait
    "Heavy bony armour: a continuous mosaic of OSTEODERMS (bony scutes) covers the back, "
    "flanks and tail, often with rows of larger keeled or spiked osteoderms. The skin "
    "between osteoderms is thick and scaly. Tank-like overall silhouette. "
    # Skull & head
    "Skull is broad, flat and triangular, with a WIDE cropping BEAK (toothless at the "
    "front). Small leaf-shaped teeth in the cheeks for grinding plants. Mouth closed or "
    "chewing. Eyes set laterally on a low broad skull. NEVER show a predator-like "
    "narrow snout with fangs. "
    # Integument
    "Thick scaly hide between the osteoderms; no feathers. "
)

_CLADE_CERATOPSIA = (
    # Pose & framing
    "The animal stands in three-quarter front view to showcase the FRILL and horns "
    "facing the camera. Quadrupedal, head lowered defensively or raised in display. "
    # Posture & body plan
    "Quadrupedal, robust and rhino-like in build, with strong shoulders, thick legs, "
    "and a short tail. "
    # Frill and horns — defining trait
    "Head ornaments: a large bony FRILL extends back from the skull over the neck "
    "(species-specific shape: solid Triceratops shield vs. fenestrated Styracosaurus). "
    "HORNS on the snout and/or above the eyes — the exact number, length and "
    "orientation come from the species-specific block. Frill edges may carry "
    "epi-occipitals (small bony spikes). "
    # Skull & head
    "Skull is enormous relative to body length (sometimes ~30% of total length), with "
    "a powerful parrot-like BEAK at the front for cropping plants. Cheek teeth are in "
    "dental batteries for shearing. Mouth closed. No fangs. "
    # Integument
    "Thick scaly hide; no feathers (except possibly sparse quills in some basal forms "
    "like Psittacosaurus — only mention if the species is basal enough). "
)

_CLADE_ORNITHOPODA = (
    # Pose & framing
    "The animal stands in three-quarter side view, on hind legs OR quadrupedal "
    "(species can switch between both gaits). Head raised, alert and calm — not "
    "aggressive. "
    # Posture & body plan
    "Facultative bipedal/quadrupedal: tall hind limbs, smaller but functional front "
    "limbs that can bear weight when slow-walking. Body horizontal when running on two "
    "legs, more level when walking on four. Long muscular tail held off the ground. "
    # Skull & head
    "Head ends in a wide horny BEAK (rhamphotheca) — duck-like in hadrosaurs, more "
    "pointed in iguanodonts. Cheek teeth in dental batteries for grinding plants; NO "
    "fangs visible. Mouth closed or chewing. Species-specific cranial crest (e.g. "
    "Parasaurolophus tube, Edmontosaurus low crest) should come from the species block. "
    # Iguanodont thumb spike (only if applicable, species block will specify)
    "Forelimb hand may have a conical thumb spike (Iguanodon and close relatives). "
    # Integument
    "Thick scaly hide; no feathers, no plumage. "
)

_CLADE_PACHYCEPHALOSAURIA = (
    # Pose & framing
    "The animal stands in three-quarter side or front view to show the domed skull. "
    "Bipedal, head lowered slightly with the dome forward. Calm or display posture, "
    "not aggressive. "
    # Posture & body plan
    "Bipedal, stocky body, short forelimbs, robust hindlimbs, thick muscular tail held "
    "behind for balance. Spine roughly horizontal. "
    # Dome — defining trait
    "Skull cap: a very thick, rounded BONY DOME crowns the top of the skull, often "
    "rimmed by small bony bumps or short knobs around the back of the head and snout. "
    "The dome is the species' iconic feature — render it prominently. "
    # Skull & head
    "Snout is short and narrow, ending in a small beak with small leaf-shaped teeth. "
    "Eyes face laterally. Mouth closed or slightly open. No predator features. "
    # Integument
    "Thick scaly hide; no feathers. "
)

_CLADE_OTHER = (
    # Minimal fallback: just trust the species-specific block to describe the body.
    "The animal is shown in a clear three-quarter view, body fully visible, in a calm "
    "stance appropriate to its biology. Anatomy, posture and any oral structures are "
    "described in the species-specific block below. "
)

_CLADE_PROMPTS: dict[str, str] = {
    "theropoda": _CLADE_THEROPODA,
    "sauropoda": _CLADE_SAUROPODA,
    "stegosauria": _CLADE_STEGOSAURIA,
    "ankylosauria": _CLADE_ANKYLOSAURIA,
    "ceratopsia": _CLADE_CERATOPSIA,
    "ornithopoda": _CLADE_ORNITHOPODA,
    "pachycephalosauria": _CLADE_PACHYCEPHALOSAURIA,
    "other": _CLADE_OTHER,
}


def _hero_prompt(factsheet: FactSheet) -> str:
    """Compose the hero prompt: universal + clade-specific + species + footer."""
    clade = (factsheet.clade or "other").strip().lower()
    clade_block = _CLADE_PROMPTS.get(clade, _CLADE_OTHER)
    species_specific = factsheet.image_prompt.strip().rstrip(".")
    return (
        f"{_UNIVERSAL_PREAMBLE}"
        f"{clade_block}"
        f"{species_specific}. "
        f"{_NO_TEXT_FOOTER}"
    )


def generate_assets(
    factsheet: FactSheet,
    out_dir: Path,
    *,
    model: str | None = None,
    steps: int | None = None,
) -> None:
    """Produce ``hero.png`` in ``out_dir`` via Gemini.

    Args:
        factsheet: Fully validated FactSheet with image_prompt, clade and
            visual_references.
        out_dir: Directory where ``hero.png`` is written.
        model: Gemini image model name. None uses GEMINI_IMAGE_MODEL env default.
        steps: Ignored — kept for signature compatibility.
    """
    del steps
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiClient()

    body_refs = _load_ref_paths(out_dir, [r.path for r in factsheet.visual_references.body])
    effective_model = model or _image_model()
    hero_bytes = client.generate_image(
        _hero_prompt(factsheet), refs=body_refs, model=effective_model
    )
    (out_dir / "hero.png").write_bytes(hero_bytes)

    # Record which model produced this image so the catalog can show provenance.
    meta = {
        "image_model": effective_model,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "image_meta.json").write_text(json.dumps(meta, indent=2))
