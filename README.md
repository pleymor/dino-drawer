# Dino Drawer

Génère une illustration scientifique réaliste à partir d'un nom d'espèce
(préhistorique ou actuelle) : recherche bibliographique, synthèse par LLM,
classification d'images par VLM, génération photoréaliste, puis publication
sur Cloudflare R2 avec un catalogue JSON.

## Installation

Prérequis :
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Une clé API Gemini : https://aistudio.google.com/apikey (Tier 1 minimum
  pour la génération d'images — free tier suffit pour les étapes texte/vision)
- Un bucket Cloudflare R2 + token API si tu veux publier les images
  (facultatif pour générer en local)

```bash
make install
# équivalent à :
#   uv venv
#   .venv/bin/python -m pip install --upgrade pip
#   uv pip install -e ".[dev]"
#   .venv/bin/python -m playwright install chromium
```

> **Note pyenv** : si ton shell route `python` via une fonction pyenv (cas
> fréquent), `source .venv/bin/activate` ne suffit pas — le Makefile invoque
> directement `.venv/bin/python` pour contourner ça.

### Configuration `.env`

Crée un `.env` à la racine du projet. Il est gitignored et auto-chargé au
démarrage de toute commande `python -m dino_drawer.*` (via `python-dotenv`).

```bash
# === Génération (obligatoire) ===
GEMINI_API_KEY=AIza...

# === Génération (optionnel — defaults dans clients/gemini.py) ===
# GEMINI_TEXT_MODEL=gemini-2.5-flash
# GEMINI_IMAGE_MODEL=gemini-2.5-flash-image

# === Publication R2 (obligatoire UNIQUEMENT pour make publish) ===
R2_BUCKET=dino-drawer
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
R2_PUBLIC_BASE_URL=https://pub-<bucket-id>.r2.dev
```

`R2_ENDPOINT` est l'endpoint S3-compatible (auth requise). `R2_PUBLIC_BASE_URL`
est l'URL publique du bucket — il faut **activer le "R2.dev subdomain"** dans
les paramètres du bucket sur le dashboard Cloudflare pour l'obtenir.

## Usage

### Générer une espèce

```bash
make run SPECIES='Tyrannosaurus rex'
# → out/tyrannosaurus-rex/final.png
```

Autres exemples :
```bash
make run SPECIES='Velociraptor mongoliensis'
make run SPECIES='Smilodon fatalis' ARGS='--skip-refs --lang en'
make run ARGS='--force-step synthesis'  # défaut SPECIES='Tyrannosaurus rex'
```

Flags utiles (via `ARGS=`) :
- `--skip-refs` — saute le scraping d'images, plus rapide pour itérer sur le texte
- `--force-step {papers,images,filter,synthesis,diffusion,compose}` — refait cette étape et les suivantes
- `--lang en` — sortie en anglais (défaut français)
- `--max-refs 20` — limite le scraping d'images

### Publier une espèce sur R2

```bash
make publish SPECIES='Tyrannosaurus rex'
# upload out/tyrannosaurus-rex/final.png → R2 sous tyrannosaurus-rex.webp
# met à jour published/catalog.json
```

`make publish-all` publie toutes les espèces générées sous `out/*/`.
`make unpublish SPECIES='...'` supprime du bucket + du catalogue.

Le fichier `published/catalog.json` est **commit-able** dans le repo. Il
contient tout le contenu textuel de chaque espèce + l'URL publique de son
image. Format :

```json
{
  "count": 2,
  "generated_at": "...",
  "species": [
    {
      "slug": "tyrannosaurus-rex",
      "species": "Tyrannosaurus rex",
      "subtitle": "...",
      "annotations": [...],
      "skull_view": {...},
      "size": {...},
      "conclusion": "...",
      "references": [...],
      "image_prompt": "...",
      "image_url": "https://pub-xxx.r2.dev/tyrannosaurus-rex.webp",
      "generated_at": "..."
    }
  ]
}
```

Ton site statique fetch ce JSON et utilise `image_url` dans un `<img>`.

### Étape par étape (debug)

```bash
make papers SPECIES='...'      # papers.json
make images SPECIES='...'      # refs_raw/ + refs_raw.json
make vision SPECIES='...'      # filtrage VLM → refs.json
make synthesis SPECIES='...'   # factsheet.json
make image SPECIES='...'       # hero.png + skull.png + silhouette.svg
make compose SPECIES='...'     # final.png (juste le hero, sans annotation)
```

## Pipeline

1. **Papers** — Semantic Scholar + OpenAlex + Wikipedia (parallèle avec étape 2)
2. **Images** — scraping Wikimedia Commons + PhyloPic + iNaturalist
3. **Filter** — Gemini multimodal classe chaque image, garde top 3 corps + top 2 crâne
4. **Synthesis** — Gemini produit `factsheet.json` (faits sourcés + prompt visuel)
5. **Generation** — Gemini Image génère `hero.png` à partir du prompt + refs visuelles
6. **Compose** — produit `final.png` (juste le hero, recadré 2000×1200, sans texte)
7. **Publish** *(optionnel)* — WebP + upload R2 + mise à jour `published/catalog.json`

Voir `docs/superpowers/specs/2026-05-12-dino-drawer-design.md` pour le spec
d'origine (note : décrivait une stack 100% locale, le projet utilise
maintenant Gemini). Le plan d'implémentation détaillé est dans
`docs/superpowers/plans/2026-05-12-dino-drawer.md`.

## Modèles Gemini

| Variable | Défaut | Tier requis | Notes |
|---|---|---|---|
| `GEMINI_TEXT_MODEL` | `gemini-2.5-flash` | Free | VLM + LLM texte |
| `GEMINI_IMAGE_MODEL` | `gemini-2.5-flash-image` | **Tier 1** (billing activé) | Génération d'image |

> **Important** : la génération d'image Gemini nécessite **billing activé**
> sur ton compte Google AI Studio. Sur le free-tier, les modèles d'image ont
> `limit: 0`. Active le billing sur https://aistudio.google.com (quelques
> dollars suffisent pour des dizaines d'espèces).

Le client retry automatiquement les 429 et 503 jusqu'à 8 fois avec un backoff
qui respecte le `retry_delay` annoncé par l'API.

## Développement

```bash
make test                    # toute la suite (mocks, aucun appel réseau ou Gemini)
.venv/bin/python -m pytest tests/test_compose.py -v
```

76 tests couvrent les modèles, la recherche, le filtrage VLM, la synthèse,
la diffusion, la composition, l'agent, le client Gemini et le publish.
