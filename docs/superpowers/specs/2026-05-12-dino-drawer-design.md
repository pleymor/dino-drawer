# Dino Drawer — Design Spec

**Date** : 2026-05-12
**Status** : Draft, en attente de revue utilisateur

## 1. Objectif

Construire un agent en ligne de commande qui, à partir du nom d'une espèce (en priorité préhistorique), produit une infographie scientifique sourcée : illustration réaliste de l'animal dans son environnement, annotations anatomiques, vue de profil du crâne, échelle comparative avec silhouette humaine, et bandeau de références scientifiques récentes.

Référence visuelle de sortie : `tyrannosaurus-rex.png` à la racine du dépôt.

**Non-objectifs (v1)** :
- Pas d'UI web ni d'API HTTP — uniquement CLI.
- Pas de variantes de layout — un seul template d'infographie.
- Pas de génération de texte par modèle de diffusion : tout le texte est rendu par le template HTML.

## 2. Contraintes fortes

- **Exécution locale des LLM** : aucun appel à OpenAI / Anthropic / Google pour la synthèse de texte. Ollama est le runtime cible.
- **Plateforme cible** : macOS Apple Silicon, ≥16 Go de RAM unifiée. Le code reste portable mais n'est pas testé sur CUDA en v1.
- **Texte hors image** : le modèle de diffusion ne produit que de l'iconographie ; le texte est composé par un template HTML/CSS.
- **Sources citables** : chaque fait anatomique de la fiche doit être rattaché à au moins une référence (DOI ou URL Wikipedia).

## 3. Architecture

Pipeline à 4 étapes orchestré par un agent Python. La fiche JSON intermédiaire est l'unique source de vérité ; chaque étape peut être relancée indépendamment.

```
nom_espèce
   │
   ▼
[1. Recherche]        ──► papers.json  (résumés + DOI + métadonnées)
   │
   ▼
[2. Synthèse LLM]     ──► factsheet.json  (fiche structurée, sources par fait)
   │
   ▼
[3. Diffusion image]  ──► hero.png, skull.png, silhouette.svg
   │
   ▼
[4. Composition]      ──► final.png
```

### 3.1 Module Recherche (`research/`)

**Responsabilité** : à partir d'un nom d'espèce, retourner une liste de publications scientifiques pertinentes des 5 dernières années, avec résumés.

**Sources interrogées** (dans cet ordre, avec dédoublonnage par DOI) :
1. **Semantic Scholar API** (`api.semanticscholar.org/graph/v1`) — recherche par mot-clé, retourne titre, auteurs, année, résumé, DOI, citations.
2. **OpenAlex API** (`api.openalex.org/works`) — fallback et complément, filtre par concept paléontologie.
3. **Wikipedia REST API** — extrait introductif + résumé des sections "Description" et "Paleobiology" comme contexte de base (toujours cité).

**Sortie** : `papers.json`
```json
{
  "species": "Tyrannosaurus rex",
  "wikipedia": {"url": "...", "extract": "..."},
  "papers": [
    {
      "doi": "10.xxxx/...",
      "title": "...",
      "authors": ["DePalma, R.A.", "..."],
      "year": 2020,
      "abstract": "...",
      "source": "semantic_scholar"
    }
  ]
}
```

**Comportement en l'absence de résultats** : si Semantic Scholar et OpenAlex retournent zéro papier (cas d'une espèce obscure), l'étape n'échoue pas — elle continue avec uniquement le contexte Wikipedia, et la fiche finale le signale.

### 3.2 Module Synthèse (`synthesis/`)

**Responsabilité** : transformer `papers.json` en `factsheet.json` structuré, avec rattachement des faits aux sources.

**Runtime** : Ollama local. Modèle par défaut : `qwen2.5:14b-instruct`. Fallback léger : `llama3.1:8b-instruct`. Configurable via variable d'env `DINO_LLM_MODEL`.

**Stratégie de prompt** : un seul appel structuré qui demande au modèle de remplir un schéma Pydantic. Validation côté Python ; en cas de violation du schéma, réessai automatique (max 2) avec le message d'erreur injecté.

**Schéma de sortie** (`factsheet.json`) :
```json
{
  "species": "Tyrannosaurus rex",
  "subtitle": "Reconstitution basée sur les données scientifiques les plus récentes (2020-2024)",
  "annotations": [
    {
      "region": "tête",
      "facts": ["Museau robuste et profond", "Lèvres recouvrant les dents..."],
      "source_ids": ["paper:0", "paper:3"]
    },
    {"region": "peau_et_couverture", "facts": [...], "source_ids": [...]},
    {"region": "membres_anterieurs", "facts": [...], "source_ids": [...]},
    {"region": "membres_posterieurs", "facts": [...], "source_ids": [...]},
    {"region": "queue", "facts": [...], "source_ids": [...]}
  ],
  "skull_view": {
    "facts": ["..."],
    "scale_cm": 50,
    "source_ids": ["..."]
  },
  "size": {
    "length_m": [12, 13],
    "hip_height_m": [3.5, 4],
    "source_ids": ["..."]
  },
  "conclusion": "Les découvertes récentes suggèrent que T. rex avait des lèvres couvrant les dents, ...",
  "references": [
    {"id": "paper:0", "citation_short": "DePalma, R.A. et al. (2020)", "doi": "...", "title": "..."}
  ],
  "image_prompt": "photorealistic adult Tyrannosaurus rex, lateral profile, dense forest of late Cretaceous Laramidia, hazy morning light, feathered patches on neck and back, lipped mouth, no text"
}
```

**Règles de remplissage** :
- Chaque entrée `facts` doit pointer vers au moins un `source_id` existant dans `references`.
- `image_prompt` est généré par le LLM avec une instruction explicite "no text, no labels, no captions, photoréaliste".
- Les régions anatomiques sont fixes (5 régions + crâne + taille), pour matcher le template. Si une région n'a aucun fait, le LLM remplit avec un fait baseline issu de Wikipedia plutôt que de laisser vide.

### 3.3 Module Image (`image/`)

**Responsabilité** : produire trois assets visuels à partir de `factsheet.json`.

**Assets** :
1. `hero.png` — illustration principale, 1600×1000, l'animal en pied dans son environnement.
2. `skull.png` — vue latérale du crâne, fond transparent ou neutre.
3. `silhouette.svg` — silhouette stylisée (générée par code, pas par diffusion) à côté d'une silhouette humaine 1,75 m, pour l'échelle.

**Runtime** : `diffusers` (Hugging Face) sur backend MPS (Apple Silicon).

**Modèle par défaut** : `black-forest-labs/FLUX.1-schnell` (4 steps, quantif fp16). Fallback : `stabilityai/stable-diffusion-xl-base-1.0`. Configurable via `DINO_IMAGE_MODEL`.

**Prompts** :
- Le prompt principal vient de `factsheet.json::image_prompt`, complété par un suffixe négatif géré côté code : `"no text, no watermark, no captions, no signature"`.
- Le prompt du crâne est dérivé par le code : `"detailed lateral view of {species} skull, scientific illustration style, neutral background, no text"`.

**Silhouette pour l'échelle** : générée par code en SVG, pas par diffusion — un trait noir simple basé sur les dimensions `size.length_m` / `size.hip_height_m`, à côté d'un humain stylisé de 1,75 m. Cela garantit une échelle exacte et un rendu net.

### 3.4 Module Composition (`compose/`)

**Responsabilité** : assembler `final.png` à partir de `factsheet.json` + les 3 assets.

**Approche** : template HTML/CSS rendu par Playwright (Chromium headless), puis `page.screenshot()` pour exporter en PNG.

**Pourquoi pas Pillow ou ReportLab** : la mise en page de l'exemple (annotations avec lignes pointant vers le corps, typographie, alignements multiples) est triviale en HTML/CSS et pénible en API de dessin impérative.

**Template** : `compose/templates/infographic.html` + `infographic.css`. Variables remplies par Jinja2.

**Sortie** : `final.png` à la résolution 2000×1200 (configurable).

### 3.5 Orchestration (`agent.py`)

Classe `DinoDrawerAgent` avec méthodes :
- `run(species: str) -> Path` — pipeline complet.
- `step_research(species)`, `step_synthesis(papers)`, `step_image(factsheet)`, `step_compose(factsheet, assets)` — étapes individuelles, idempotentes.

**Mise en cache** : chaque étape écrit son output dans `out/<slug-espèce>/`. Si le fichier existe, l'étape est sautée sauf si `--force` ou `--force-step <nom>` est passé.

## 4. Structure du dépôt

```
dino-drawer/
├── pyproject.toml
├── README.md
├── tyrannosaurus-rex.png            # référence visuelle d'origine
├── docs/superpowers/specs/          # ce document et les suivants
├── src/dino_drawer/
│   ├── __init__.py
│   ├── __main__.py                  # entry point CLI
│   ├── agent.py                     # orchestrateur
│   ├── models.py                    # schémas Pydantic
│   ├── research/
│   │   ├── semantic_scholar.py
│   │   ├── openalex.py
│   │   └── wikipedia.py
│   ├── synthesis/
│   │   ├── ollama_client.py
│   │   └── prompts.py
│   ├── image/
│   │   ├── diffusion.py
│   │   └── silhouette.py            # SVG par code
│   └── compose/
│       ├── render.py                # Playwright
│       └── templates/
│           ├── infographic.html
│           └── infographic.css
├── tests/
│   ├── test_research.py
│   ├── test_synthesis.py
│   ├── test_image.py
│   ├── test_compose.py
│   └── fixtures/                    # papers.json et factsheet.json figés
└── out/                             # généré, .gitignore
```

## 5. Interfaces

### 5.1 CLI

```
python -m dino_drawer "Tyrannosaurus rex"
python -m dino_drawer "Triceratops horridus" --out ./out --force-step image
python -m dino_drawer --help
```

**Flags** :
- `--out PATH` (défaut : `./out`)
- `--force` — refait tout
- `--force-step {research,synthesis,image,compose}` — refait une étape (et tout ce qui suit)
- `--model-llm STR` (défaut : `qwen2.5:14b-instruct`)
- `--model-image STR` (défaut : `black-forest-labs/FLUX.1-schnell`)
- `--lang {fr,en}` (défaut : `fr`)

### 5.2 Variables d'environnement

- `DINO_LLM_MODEL` — override du modèle Ollama
- `DINO_IMAGE_MODEL` — override du modèle de diffusion
- `OLLAMA_HOST` — défaut `http://localhost:11434`
- `HF_HOME` — cache Hugging Face

## 6. Gestion d'erreurs

- **Ollama indisponible** : message clair "démarre Ollama et installe le modèle X avec `ollama pull X`", code de sortie 2.
- **Modèle de diffusion non téléchargé** : tentative de téléchargement automatique avec barre de progression ; si échec réseau, message clair.
- **Aucun papier trouvé** : warning, on continue avec Wikipedia seul, le bandeau de références dit "Sources : Wikipedia" et la conclusion mentionne le manque de littérature récente.
- **LLM produit un JSON invalide** : 2 réessais avec l'erreur de validation injectée dans le prompt. Au-delà, on échoue avec le dernier output brut sauvegardé pour debug.
- **Playwright manquant** : message d'install `playwright install chromium`.

## 7. Tests

Approche TDD, conformément aux instructions globales utilisateur.

**Tests unitaires** :
- `research/` : mocks HTTP avec `respx` ou fixtures JSON enregistrées (pas d'appels réseau en CI).
- `synthesis/` : test du parsing strict du schéma Pydantic, test du retry sur JSON invalide (mock Ollama).
- `image/silhouette.py` : test que le SVG généré contient les bonnes proportions.
- `compose/render.py` : test que le HTML rendu contient bien toutes les sections (snapshot DOM, pas pixel-perfect).

**Tests d'intégration** :
- Un test bout-en-bout avec une fiche factice (pas d'appel LLM, pas d'appel diffusion) qui vérifie que `compose` produit un PNG aux bonnes dimensions.
- Pas de test qui appelle réellement Ollama ou diffusion en CI — trop lents, dépendants de l'hôte.

**Fixtures** : `tests/fixtures/trex_papers.json` et `tests/fixtures/trex_factsheet.json` figées et versionnées.

## 8. Décisions techniques explicites

| Question | Choix | Raison |
|---|---|---|
| Python vs TypeScript | Python | Écosystème AI (diffusers, ollama-python) plus mûr |
| Ollama vs llama.cpp direct | Ollama | Gestion de modèles, API HTTP stable, change de modèle sans recoder |
| Flux.1-schnell vs SDXL | Flux par défaut | Meilleure qualité photo en 4 steps, suffisant en 16-24 Go RAM unifiée |
| Diffusers vs ComfyUI | Diffusers | Pas de serveur séparé, intégration Python directe ; ComfyUI envisageable plus tard |
| HTML/CSS vs Pillow | HTML/CSS + Playwright | Typographie nette, mise en page complexe triviale |
| LangGraph vs agent custom | Custom | Pipeline linéaire, pas de boucles d'outils ; LangGraph est overkill |
| Pydantic v1 vs v2 | v2 | Standard actuel, meilleure perf et messages d'erreur |
| Jinja2 vs f-strings | Jinja2 | Template HTML séparé du code, plus maintenable |

## 9. Hors-périmètre v1 (à envisager plus tard)

- Génération multi-langues simultanée (FR + EN sur une même image).
- Support de plusieurs templates de layout (portrait, paysage, format réseaux sociaux).
- UI web pour itérer sur prompts visuels.
- Cache de la recherche scientifique partagé entre espèces.
- Support GPU NVIDIA testé.
- Animations / GIF / vidéo de l'animal.
- Validation paléontologique automatique (cross-check entre papiers).

## 10. Critères de succès v1

1. `python -m dino_drawer "Tyrannosaurus rex"` produit un PNG dont le contenu et la mise en page sont du même niveau de qualité que `tyrannosaurus-rex.png`.
2. Le même pipeline produit un résultat raisonnable pour au moins 3 autres espèces : `Triceratops horridus`, `Velociraptor mongoliensis`, `Smilodon fatalis`.
3. Chaque fait textuel de la fiche est rattaché à une référence présente dans le bandeau du bas.
4. Toute la synthèse de texte tourne en local via Ollama, sans appel à des LLM hébergés.
5. Temps total d'exécution < 5 minutes par espèce sur Mac M-series, après que les modèles soient téléchargés.
