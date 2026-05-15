---
name: expand-dino-catalog
description: Find dinosaur species listed on lifetree.pleymor.com but not yet in the local published/catalog.json, then run the dino-drawer pipeline in parallel batches of 3 to generate hero images for them. Stop when done, on rate-limit, or on critical error. The user reviews and publishes manually.
---

# Expand the dino catalog

Trigger when the user wants to bulk-generate hero images for dinosaur species that are still missing from the local catalog. Typical asks: "fais autant de dinos que possible", "complète le catalogue", "lance tous les dinos manquants", "populate the catalog from the lifetree".

The pipeline costs real Gemini tokens (synthesis + image gen per species). Always present the scope and get explicit user confirmation before launching more than ~3 species in a row.

## Workflow

### 1. Build the "missing species" list

a. **Read the existing catalog.** Get the set of slugs already published:
   ```python
   import json
   cat = json.load(open("published/catalog.json"))
   existing = {entry["slug"] for entry in cat["species"]}
   ```
   Slug format is `genus-species`, lowercase, hyphen-separated.

b. **Scrape the lifetree taxonomy.** Source URL:
   `https://lifetree.pleymor.com/#bilateria/deuterostomia/chordata/vertebrata/gnathostomata/tetrapoda/amniota/sauropsida/archosauria/dinosauria`

   The page uses a URL hash fragment for routing (SPA). Try in order of preference:
   1. `WebFetch` on the URL — if the static HTML or a discoverable JSON endpoint (e.g. `https://lifetree.pleymor.com/data/...json`) gives the species data, use that.
   2. If JS-only, use `mcp__chrome-devtools__navigate_page` + `take_snapshot` to inspect the rendered DOM.
   3. Look in the page source for inline `<script>` data or fetch calls — there might be a single JSON blob loaded at startup.

   The tree is **nested**: dinosauria contains sub-groups (Theropoda, Sauropodomorpha, Ornithischia, etc.), each containing more sub-groups, with leaf species at the bottom. You MUST recursively descend to collect every leaf binomial. Leaves are typically two-word Latin names like "Tyrannosaurus rex", "Carnotaurus sastrei". Skip non-binomial labels (clade names, "incertae sedis", "Aves" branch, vernacular names).

c. **Compute the diff:**
   ```python
   def slugify(species: str) -> str:
       return species.lower().replace(" ", "-")
   missing = [sp for sp in lifetree_species if slugify(sp) not in existing]
   ```

d. **Present the scope to the user.** Show the count and a sample (e.g., "Found 47 species missing. First 10: Allosaurus fragilis, Carnotaurus sastrei, ...). Estimate wall time (≈ 6-8 min per species when running 3-parallel → 47 / 3 × 7 min ≈ 110 min). Ask explicit confirmation before launching.

### 2. Run the pipeline in parallel batches of 3

For each batch of up to 3 species from `missing`:

```bash
make run-batch SPECIES_LIST='Species A|Species B|Species C'
```

Or the equivalent direct command (if `make` is unavailable):
```bash
for sp in "Species A" "Species B" "Species C"; do
  slug=$(echo "$sp" | tr '[:upper:] ' '[:lower:]-')
  .venv/bin/python -m dino_drawer "$sp" > "out/$slug.log" 2>&1 &
done
wait
```

Run each batch as a single foreground command (don't kick off the next batch until the current one finishes). Use `run_in_background: true` on the bash call so Claude isn't blocked, and rely on the harness completion notification.

After each batch:

- For each species in the batch, check `out/<slug>/final.png` exists. If yes → success. If no → tail the log to see what failed.
- If any species failed, log the species name + a one-line error summary and continue to the next batch (don't retry within the loop).

### 3. Rate-limit handling

If a log shows a 429 / rate-limit / quota error from Gemini (`GeminiError`, `RESOURCE_EXHAUSTED`, etc.):
- Stop launching new batches.
- Wait ~60 seconds.
- Resume with the species that failed (add it back to the queue).
- If 429 happens again in the next batch, stop entirely and tell the user.

### 4. Don't auto-publish

The pipeline produces `out/<slug>/final.png`. Do NOT run `make publish` or `python -m dino_drawer.publish` for any new species. The user reviews all generated images before deciding which to publish.

### 5. Final report

When the loop ends (all species done, rate-limited, or stopped):
- List of slugs that produced a `final.png` (ready for review).
- List of slugs that failed, with a one-line cause.
- A reminder: "Run `make publish-all` once you've reviewed and approved the images, or `make publish SPECIES='X'` per species."

## Edge cases

- **Latin binomial only.** Skip entries that aren't `Genus species` (skip "Tyrannosauridae", "Coelurosauria incertae sedis", common names in parentheses, etc.). Strip qualifiers like `"Tyrannosaurus rex (juvenile)"` → `"Tyrannosaurus rex"` before slugifying.
- **Stop on critical errors.** If the venv is missing, papers API is dead, or the catalog can't be parsed, stop immediately and report. Don't bash forward.
- **Don't re-run existing species.** The pipeline has skip-if-exists logic per step, so re-launching is mostly idempotent — but a partially-completed `out/<slug>/` will resume from where it stopped. If the user wants a fresh re-run, they should use `--force` themselves.
- **Confirm before very large batches.** If `missing` contains more than ~20 species, get explicit user confirmation about the cost/time before launching.

## Files this skill touches

- Reads: `published/catalog.json`
- Writes: `out/<slug>/*` (artifacts) and `out/<slug>.log` (per-species log)
- Network: lifetree.pleymor.com + Gemini API (via the pipeline)
