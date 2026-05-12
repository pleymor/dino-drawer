# Dino Drawer

Génère une infographie scientifique sourcée à partir d'un nom d'espèce
(préhistorique ou actuelle). Tout tourne en local : recherche bibliographique,
synthèse par LLM, classification d'images par VLM, diffusion conditionnée
par IP-Adapter, et composition HTML/CSS.

## Installation

Prérequis :
- macOS Apple Silicon (16 Go RAM minimum, 24 Go recommandé)
- Python 3.11+
- [Ollama](https://ollama.com)

Une fois cloné :
```bash
ollama pull qwen2.5:14b-instruct
ollama pull qwen2.5vl:7b
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
playwright install chromium
```

## Usage

```bash
python -m dino_drawer "Tyrannosaurus rex"
# → out/tyrannosaurus-rex/final.png
```

Flags utiles :
- `--skip-refs` : saute le scraping d'images, plus rapide pour itérer sur le texte.
- `--force-step diffusion` : re-génère uniquement l'image et la composition.
- `--lang en` : sortie en anglais.
- `--max-refs 20` : limite le scraping.

## Pipeline

1. **Papers** — Semantic Scholar + OpenAlex + Wikipedia (parallèle avec étape 2).
2. **Images** — Wikimedia Commons + PhyloPic + iNaturalist.
3. **Filter** — VLM (Qwen2.5-VL) classe les images et garde le top 3 corps / top 2 crâne.
4. **Synthesis** — LLM (Qwen2.5) produit `factsheet.json` enrichi par les descriptions VLM des refs.
5. **Diffusion** — SDXL + IP-Adapter Plus conditionne la génération sur les refs sélectionnées.
6. **Compose** — Template HTML/CSS rendu par Playwright pour le `final.png`.

Voir `docs/superpowers/specs/2026-05-12-dino-drawer-design.md` pour le détail.

## Développement

```bash
pytest                       # tous les tests (mocks, pas d'appels réseau ni LLM)
pytest tests/test_compose.py # un seul module
```
