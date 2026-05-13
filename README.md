# Dino Drawer

Génère une infographie scientifique sourcée à partir d'un nom d'espèce
(préhistorique ou actuelle) : recherche bibliographique, synthèse par LLM,
classification d'images par VLM, génération d'illustration et composition
HTML/CSS.

## Installation

Prérequis :
- Python 3.11+
- Une clé API Gemini gratuite : https://aistudio.google.com/apikey

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
playwright install chromium
```

Crée un fichier `.env` à la racine du projet :
```
GEMINI_API_KEY=AIza...
```

`.env` est gitignored. Il est auto-chargé au démarrage de toute commande
`python -m dino_drawer.*`.

### Configuration (optionnelle)

Variables d'env reconnues, valeurs par défaut adaptées au free-tier Gemini :

- `GEMINI_TEXT_MODEL` — défaut `gemini-2.5-flash` (synthèse + VLM)
- `GEMINI_IMAGE_MODEL` — défaut `gemini-3-pro-image-preview`

## Usage

```bash
python -m dino_drawer "Tyrannosaurus rex"
# → out/tyrannosaurus-rex/final.png
```

Flags utiles :
- `--skip-refs` : saute le scraping d'images, plus rapide pour itérer sur le texte.
- `--force-step diffusion` : re-génère uniquement l'image et la composition.
- `--force-step synthesis` : re-fait synthèse + diffusion + compose.
- `--lang en` : sortie en anglais.
- `--max-refs 20` : limite le scraping d'images.

### Étape par étape (debug)

```bash
python -m dino_drawer.research.papers "Tyrannosaurus rex"   # papers.json
python -m dino_drawer.research.images "Tyrannosaurus rex"   # refs_raw/
python -m dino_drawer.vision out/tyrannosaurus-rex          # refs.json
python -m dino_drawer.synthesis out/tyrannosaurus-rex       # factsheet.json
python -m dino_drawer.image out/tyrannosaurus-rex           # hero.png + skull.png
python -m dino_drawer.compose out/tyrannosaurus-rex         # final.png
```

## Pipeline

1. **Papers** — Semantic Scholar + OpenAlex + Wikipedia (parallèle avec étape 2).
2. **Images** — scraping Wikimedia Commons + PhyloPic + iNaturalist.
3. **Filter** — Gemini multimodal classe chaque image et garde le top 3 corps / top 2 crâne.
4. **Synthesis** — Gemini produit `factsheet.json` enrichi par les descriptions VLM des refs.
5. **Generation** — Gemini 3 Pro Image génère hero.png et skull.png à partir du prompt + refs.
6. **Compose** — Template HTML/CSS rendu par Playwright pour le `final.png`.

Voir `docs/superpowers/specs/2026-05-12-dino-drawer-design.md` pour le détail
(note : le spec décrit l'architecture initiale 100% locale ; le projet est
maintenant basé sur Gemini).

## Free-tier Gemini

Pour `gemini-2.5-flash` : 10 RPM, 250 req/jour. Le filtrage VLM appelle l'API
~40 fois (une par image scrappée). Avec retry-on-429 et backoff exponentiel,
le pipeline gère ces limites automatiquement, mais peut prendre 4-5 min sur
l'étape de filtrage si les limites sont atteintes.

`gemini-3-pro-image-preview` : limites plus serrées, vérifie ton quota sur
https://aistudio.google.com avant de lancer beaucoup d'espèces.

## Développement

```bash
pytest                       # tous les tests (mocks, pas d'appels réseau ni Gemini)
pytest tests/test_compose.py # un seul module
```
