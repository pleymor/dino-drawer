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

- **Exécution locale des LLM / VLM** : aucun appel à OpenAI / Anthropic / Google pour la synthèse de texte ou l'analyse d'image. Ollama est le runtime cible.
- **Plateforme cible** : macOS Apple Silicon, ≥16 Go de RAM unifiée. Le code reste portable mais n'est pas testé sur CUDA en v1.
- **Texte hors image** : le modèle de diffusion ne produit que de l'iconographie ; le texte est composé par un template HTML/CSS.
- **Sources citables** : chaque fait anatomique de la fiche doit être rattaché à au moins une référence (DOI ou URL Wikipedia).
- **Licences d'images** : seules les images sous licence claire (Public Domain, CC-*, ou équivalent) sont téléchargées. Crédit et licence stockés à côté de chaque image.

## 3. Architecture

Pipeline à 6 étapes orchestré par un agent Python. La fiche JSON intermédiaire est l'unique source de vérité ; chaque étape peut être relancée indépendamment.

```
nom_espèce
   │
   ├──► [1. Recherche papiers]        ──► papers.json
   │
   ├──► [2. Scrape images réf.]       ──► refs_raw/  +  refs_raw.json
   │
   ▼
[3. Filtrage VLM]                     ──► refs/ + refs.json
   │   classe chaque image, garde corps + crâne utilisables
   │
   ▼
[4. Synthèse LLM + VLM]               ──► factsheet.json
   │   texte sourcé, image_prompt enrichi par description VLM des refs
   │
   ▼
[5. Diffusion conditionnée]           ──► hero.png, skull.png, silhouette.svg
   │   IP-Adapter sur les meilleures refs
   │
   ▼
[6. Composition]                      ──► final.png
```

Les étapes 1 et 2 sont indépendantes et peuvent s'exécuter en parallèle.

### 3.1 Module Recherche papiers (`research/papers/`)

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

### 3.2 Module Scrape images (`research/images/`)

**Responsabilité** : récupérer 20 à 50 images de l'espèce depuis des sources à licence propre, avec leurs métadonnées.

**Sources interrogées** :

| Source | Couverture | API |
|---|---|---|
| **Wikimedia Commons** | Paleoart, squelettes, fossiles, photos de spécimens | `commons.wikimedia.org/w/api.php` (action=query, generator=categorymembers + search) |
| **Wikipedia article images** | Galerie de l'article espèce | REST API + parsing des thumbnails |
| **PhyloPic** | Silhouettes vectorielles propres (utiles pour l'échelle) | `api.phylopic.org` |
| **iNaturalist** | Photos haute qualité (espèces actuelles uniquement) | `api.inaturalist.org/v1/taxa` + `/observations` |

**Stratégie de requête** :
- Recherche sur le nom binomial (`Tyrannosaurus rex`) **et** sur le genre seul (`Tyrannosaurus`) pour ratisser large.
- Pour Wikimedia, prioriser les catégories de type `Paleoart of <species>` et `<species>` quand elles existent.
- Limite : 50 images max par espèce, taille minimum 800×600.

**Filtrage initial (avant VLM)** : rejet par taille (< 800×600), par extension (uniquement PNG, JPG, WEBP), et par mots-clés évidents dans le titre du fichier (rejet de `chart`, `phylogeny`, `cladogram`, `range_map`).

**Sortie** : `out/<espèce>/refs_raw/<n>.{ext}` + `refs_raw.json` :
```json
{
  "species": "Tyrannosaurus rex",
  "images": [
    {
      "id": 0,
      "path": "refs_raw/0.jpg",
      "source_url": "https://commons.wikimedia.org/wiki/File:...",
      "source": "wikimedia_commons",
      "credit": "John Conway",
      "license": "CC-BY-SA-4.0",
      "width": 1920,
      "height": 1280,
      "search_query": "Tyrannosaurus rex paleoart"
    }
  ]
}
```

**Comportement en l'absence de résultats** : warning, on saute les étapes 3 (filtrage) et le conditioning de l'étape 5 — la diffusion tourne en mode "texte seul" comme initialement prévu.

### 3.3 Module Filtrage VLM (`vision/`)

**Responsabilité** : classer chaque image de `refs_raw/` et sélectionner les meilleures pour conditionner la diffusion.

**Runtime** : Ollama, modèle `qwen2.5vl:7b` par défaut (rapide, multilingue, support multimodal stable). Fallback : `llama3.2-vision:11b`. Configurable via `DINO_VLM_MODEL`.

**Prompt** (structuré, sortie JSON validée par Pydantic) :

```
Tu analyses une image candidate pour servir de référence à la génération
d'illustration scientifique de Tyrannosaurus rex.

Renvoie un JSON :
{
  "type": "paleoart_realiste" | "rendu_3d" | "photo_squelette" | "fossile" |
          "schema_anatomique" | "cladogramme" | "illustration_enfant" |
          "photo_specimen_vivant" | "carte_distribution" | "autre",
  "view": "profil_corps" | "trois_quarts_corps" | "face_corps" |
          "crane_profil" | "crane_face" | "detail" | "scene_groupe" | "autre",
  "usable_for_body_generation": bool,
  "usable_for_skull_generation": bool,
  "realism_score": 0-10,
  "quality_score": 0-10,
  "description_courte": "string"
}
```

**Règle de sélection** :
- `usable_for_body_generation` : type ∈ {paleoart_realiste, rendu_3d, photo_specimen_vivant} **et** view ∈ {profil_corps, trois_quarts_corps} **et** realism_score ≥ 6.
- `usable_for_skull_generation` : view ∈ {crane_profil, crane_face} **et** quality_score ≥ 5 (squelettes acceptés ici car utiles pour le crâne).
- Sélection finale : top 3 pour le corps (tri par `realism_score * quality_score`), top 2 pour le crâne.
- Rejet systématique : type ∈ {cladogramme, illustration_enfant, carte_distribution}.

**Sortie** : copies des images retenues dans `out/<espèce>/refs/body_<n>.<ext>` et `refs/skull_<n>.<ext>` + `refs.json` enrichi (classification + scores).

### 3.4 Module Synthèse LLM (`synthesis/`)

**Responsabilité** : transformer `papers.json` + `refs.json` en `factsheet.json` structuré, avec rattachement des faits aux sources et un `image_prompt` enrichi par l'analyse visuelle.

**Runtime** : Ollama local. Modèle texte par défaut : `qwen2.5:14b-instruct`. Fallback léger : `llama3.1:8b-instruct`. Configurable via `DINO_LLM_MODEL`.

**Deux sous-étapes** :

1. **Description VLM des références retenues** : pour chacune des 3 meilleures refs corps, on appelle le VLM avec le prompt `"Décris cette image en 2 phrases : couleurs et motifs de la peau, présence et localisation des plumes/poils/écailles, posture, environnement visible. Ne mentionne pas l'espèce."` → on accumule ces 3 descriptions dans `visual_brief`.

2. **Synthèse texte** : appel LLM avec `papers.json`, `wikipedia`, et `visual_brief` en contexte. Demande la fiche complète + un `image_prompt` qui intègre les éléments visuels observés.

**Stratégie** : validation Pydantic stricte, retry max 2 sur JSON invalide.

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
  "visual_references": {
    "body": [
      {"path": "refs/body_0.jpg", "credit": "John Conway", "license": "CC-BY-SA-4.0", "score": 8.5}
    ],
    "skull": [
      {"path": "refs/skull_0.jpg", "credit": "...", "license": "...", "score": 7.0}
    ]
  },
  "image_prompt": "photorealistic adult Tyrannosaurus rex in lateral profile, dense Late Cretaceous Laramidia forest, hazy morning light, dark olive-brown skin with paler underside, sparse feathered patches on neck and dorsal ridge, lipped mouth concealing teeth, muscular hindlimbs, no text"
}
```

**Règles de remplissage** :
- Chaque entrée `facts` doit pointer vers au moins un `source_id` existant dans `references`.
- `image_prompt` est généré par le LLM avec contrainte explicite "no text, no labels, no captions, photorealistic" et doit intégrer au moins 3 éléments observés dans `visual_brief`.
- Les régions anatomiques sont fixes (5 régions + crâne + taille), pour matcher le template. Si une région n'a aucun fait, le LLM remplit avec un fait baseline issu de Wikipedia plutôt que de laisser vide.

### 3.5 Module Image (`image/`)

**Responsabilité** : produire trois assets visuels à partir de `factsheet.json`.

**Assets** :
1. `hero.png` — illustration principale, 1600×1000, l'animal en pied dans son environnement.
2. `skull.png` — vue latérale du crâne, fond transparent ou neutre.
3. `silhouette.svg` — silhouette stylisée (générée par code, pas par diffusion) à côté d'une silhouette humaine 1,75 m, pour l'échelle.

**Runtime** : `diffusers` (Hugging Face) sur backend MPS (Apple Silicon).

**Modèle par défaut** : `stabilityai/stable-diffusion-xl-base-1.0` + **IP-Adapter Plus** (`h94/IP-Adapter`, variant `sdxl_models/ip-adapter-plus_sdxl_vit-h`). Choix motivé par la maturité du combo SDXL + IP-Adapter et son fonctionnement stable sur 16 Go de RAM unifiée. Configurable via `DINO_IMAGE_MODEL` (alternatives : `black-forest-labs/FLUX.1-schnell` sans IP-Adapter, ou `FLUX.1-dev` + XLabs IP-Adapter sur machines ≥24 Go).

**Conditionnement** :
- `hero.png` : prompt texte (`factsheet.image_prompt`) + IP-Adapter avec les 1 à 3 images de `visual_references.body` (poids IP-Adapter ~ 0.6, ajustable).
- `skull.png` : prompt `"detailed lateral view of {species} skull, scientific illustration style, neutral background, no text"` + IP-Adapter avec `visual_references.skull` (poids ~ 0.7, plus structurel).
- Suffixe négatif géré côté code : `"text, watermark, captions, signature, logo, multiple animals, deformed anatomy"`.

**Mode dégradé** : si `visual_references.body` est vide (étape 2/3 sans résultat), IP-Adapter est désactivé et la diffusion tourne sur prompt texte seul.

**Silhouette pour l'échelle** : générée par code en SVG, pas par diffusion — un trait noir simple basé sur les dimensions `size.length_m` / `size.hip_height_m`, à côté d'un humain stylisé de 1,75 m. Cela garantit une échelle exacte et un rendu net.

### 3.6 Module Composition (`compose/`)

**Responsabilité** : assembler `final.png` à partir de `factsheet.json` + les 3 assets.

**Approche** : template HTML/CSS rendu par Playwright (Chromium headless), puis `page.screenshot()` pour exporter en PNG.

**Pourquoi pas Pillow ou ReportLab** : la mise en page de l'exemple (annotations avec lignes pointant vers le corps, typographie, alignements multiples) est triviale en HTML/CSS et pénible en API de dessin impérative.

**Template** : `compose/templates/infographic.html` + `infographic.css`. Variables remplies par Jinja2.

**Sortie** : `final.png` à la résolution 2000×1200 (configurable).

### 3.7 Orchestration (`agent.py`)

Classe `DinoDrawerAgent` avec méthodes :
- `run(species: str) -> Path` — pipeline complet.
- `step_papers(species)`, `step_images(species)`, `step_filter(refs_raw)`, `step_synthesis(papers, refs)`, `step_diffusion(factsheet)`, `step_compose(factsheet, assets)` — étapes individuelles, idempotentes.

**Mise en cache** : chaque étape écrit son output dans `out/<slug-espèce>/`. Si le fichier de sortie attendu existe, l'étape est sautée sauf si `--force` ou `--force-step <nom>` est passé. Les étapes 1 (papers) et 2 (images) sont lancées en parallèle via `asyncio.gather`.

## 4. Structure du dépôt

```
dino-drawer/
├── pyproject.toml
├── README.md
├── tyrannosaurus-rex.png            # référence visuelle d'origine
├── docs/superpowers/specs/
├── src/dino_drawer/
│   ├── __init__.py
│   ├── __main__.py                  # entry point CLI
│   ├── agent.py                     # orchestrateur
│   ├── models.py                    # schémas Pydantic
│   ├── research/
│   │   ├── papers/
│   │   │   ├── semantic_scholar.py
│   │   │   ├── openalex.py
│   │   │   └── wikipedia.py
│   │   └── images/
│   │       ├── wikimedia.py
│   │       ├── phylopic.py
│   │       └── inaturalist.py
│   ├── vision/
│   │   ├── vlm_client.py            # client Ollama vision
│   │   ├── classifier.py            # filtrage + sélection
│   │   └── describer.py             # descriptions VLM des refs retenues
│   ├── synthesis/
│   │   ├── ollama_client.py
│   │   └── prompts.py
│   ├── image/
│   │   ├── diffusion.py             # SDXL + IP-Adapter
│   │   └── silhouette.py            # SVG par code
│   └── compose/
│       ├── render.py                # Playwright
│       └── templates/
│           ├── infographic.html
│           └── infographic.css
├── tests/
│   ├── test_research_papers.py
│   ├── test_research_images.py
│   ├── test_vision.py
│   ├── test_synthesis.py
│   ├── test_image.py
│   ├── test_compose.py
│   └── fixtures/                    # JSON figés + 5-10 images de test
└── out/                             # généré, .gitignore
```

## 5. Interfaces

### 5.1 CLI

```
python -m dino_drawer "Tyrannosaurus rex"
python -m dino_drawer "Triceratops horridus" --out ./out --force-step diffusion
python -m dino_drawer --help
```

**Flags** :
- `--out PATH` (défaut : `./out`)
- `--force` — refait tout
- `--force-step {papers,images,filter,synthesis,diffusion,compose}` — refait une étape (et tout ce qui suit)
- `--skip-refs` — saute le scraping + filtrage (mode texte seul, plus rapide pour itérer)
- `--model-llm STR` (défaut : `qwen2.5:14b-instruct`)
- `--model-vlm STR` (défaut : `qwen2.5vl:7b`)
- `--model-image STR` (défaut : `stabilityai/stable-diffusion-xl-base-1.0`)
- `--max-refs N` (défaut : 50) — plafond du scraping
- `--lang {fr,en}` (défaut : `fr`)

### 5.2 Variables d'environnement

- `DINO_LLM_MODEL` — override du modèle texte
- `DINO_VLM_MODEL` — override du modèle vision
- `DINO_IMAGE_MODEL` — override du modèle de diffusion
- `OLLAMA_HOST` — défaut `http://localhost:11434`
- `HF_HOME` — cache Hugging Face

## 6. Gestion d'erreurs

- **Ollama indisponible** : message clair "démarre Ollama et installe le modèle X avec `ollama pull X`", code de sortie 2.
- **Modèle texte ou vision non installé dans Ollama** : message d'install précis, code de sortie 2.
- **Modèle de diffusion non téléchargé** : tentative de téléchargement automatique avec barre de progression ; si échec réseau, message clair.
- **IP-Adapter weights non téléchargés** : pareil, download auto au premier run.
- **Aucun papier trouvé** : warning, on continue avec Wikipedia seul, le bandeau de références dit "Sources : Wikipedia" et la conclusion mentionne le manque de littérature récente.
- **Aucune image trouvée / aucune utilisable après filtrage** : warning, on bascule la diffusion en mode texte seul, et le rapport final mentionne "Pas de référence visuelle utilisée".
- **LLM ou VLM produit un JSON invalide** : 2 réessais avec l'erreur de validation injectée dans le prompt. Au-delà, on échoue avec le dernier output brut sauvegardé pour debug.
- **Playwright manquant** : message d'install `playwright install chromium`.
- **Image source téléchargée mais corrompue** : skip silencieux, log.

## 7. Tests

Approche TDD, conformément aux instructions globales utilisateur.

**Tests unitaires** :
- `research/papers/` : mocks HTTP avec `respx` ou fixtures JSON enregistrées (pas d'appels réseau en CI).
- `research/images/` : mocks HTTP, vérifie que les filtres taille/extension/mots-clés rejettent bien les cas attendus.
- `vision/classifier.py` : test que la règle de sélection prend les bonnes images depuis un `refs_raw.json` fixture, sans appeler le VLM (on injecte les classifications).
- `synthesis/` : test du parsing strict du schéma Pydantic, test du retry sur JSON invalide (mock Ollama). Test que `image_prompt` mentionne au moins un élément de `visual_brief` (injecté).
- `image/silhouette.py` : test que le SVG généré contient les bonnes proportions.
- `image/diffusion.py` : test que le pipeline est instancié avec ou sans IP-Adapter selon la présence de `visual_references`, sans réellement générer (mock du pipeline).
- `compose/render.py` : test que le HTML rendu contient bien toutes les sections (snapshot DOM, pas pixel-perfect).

**Tests d'intégration** :
- Un test bout-en-bout avec une fiche factice (pas d'appel LLM, pas d'appel VLM, pas d'appel diffusion) qui vérifie que `compose` produit un PNG aux bonnes dimensions.
- Pas de test qui appelle réellement Ollama, le VLM ou la diffusion en CI — trop lents, dépendants de l'hôte.

**Fixtures** : `tests/fixtures/trex_papers.json`, `trex_refs_raw.json`, `trex_refs_classified.json`, `trex_factsheet.json` + 5-10 petites images PNG (squelette, paleoart, schéma) pour les tests de filtrage.

## 8. Décisions techniques explicites

| Question | Choix | Raison |
|---|---|---|
| Python vs TypeScript | Python | Écosystème AI (diffusers, ollama-python) plus mûr |
| Ollama vs llama.cpp direct | Ollama | Gestion de modèles, API HTTP stable, change de modèle sans recoder |
| Modèle image : Flux vs SDXL | **SDXL + IP-Adapter Plus** | Combo mature, IP-Adapter stable, tient en 16 Go RAM unifiée. Flux disponible en option mais sans IP-Adapter par défaut. |
| VLM : Qwen2.5-VL vs LLaVA | `qwen2.5vl:7b` | Multilingue, sortie JSON fiable, ~6 Go RAM |
| Conditionnement : IP-Adapter vs ControlNet | IP-Adapter (v1) | Subject-conditioning suffit pour la cohérence visuelle ; ControlNet (depth/pose) envisageable plus tard pour figer la pose |
| Diffusers vs ComfyUI | Diffusers | Pas de serveur séparé, intégration Python directe ; ComfyUI envisageable plus tard |
| HTML/CSS vs Pillow | HTML/CSS + Playwright | Typographie nette, mise en page complexe triviale |
| LangGraph vs agent custom | Custom | Pipeline linéaire, pas de boucles d'outils ; LangGraph est overkill |
| Pydantic v1 vs v2 | v2 | Standard actuel, meilleure perf et messages d'erreur |
| Jinja2 vs f-strings | Jinja2 | Template HTML séparé du code, plus maintenable |
| Sources d'images | Wikimedia + Wikipedia + PhyloPic + iNaturalist | Licences propres, métadonnées riches. Scraping large (DDG, Bing) exclu en v1 pour des raisons de licence. |
| Étapes 1 & 2 parallèles | `asyncio.gather` | Recherche papiers et scraping images indépendants, économie de temps |

## 9. Hors-périmètre v1 (à envisager plus tard)

- Génération multi-langues simultanée (FR + EN sur une même image).
- Support de plusieurs templates de layout (portrait, paysage, format réseaux sociaux).
- UI web pour itérer sur prompts visuels.
- Cache de la recherche scientifique partagé entre espèces.
- Support GPU NVIDIA testé.
- ControlNet (depth ou pose) en plus de IP-Adapter pour figer la composition.
- Fine-tuning LoRA par clade (théropodes, sauropodes, etc.).
- Animations / GIF / vidéo de l'animal.
- Validation paléontologique automatique (cross-check entre papiers).
- Scraping étendu (DuckDuckGo, Bing) avec gestion explicite des licences.

## 10. Critères de succès v1

1. `python -m dino_drawer "Tyrannosaurus rex"` produit un PNG dont le contenu et la mise en page sont du même niveau de qualité que `tyrannosaurus-rex.png`.
2. Le même pipeline produit un résultat raisonnable pour au moins 3 autres espèces : `Triceratops horridus`, `Velociraptor mongoliensis`, `Smilodon fatalis`.
3. Pour chaque espèce, au moins 2 images de référence corps **et** 1 image de crâne sont retenues après filtrage VLM (sauf espèce très obscure, mode dégradé).
4. Chaque fait textuel de la fiche est rattaché à une référence présente dans le bandeau du bas.
5. `image_prompt` mentionne au moins 3 éléments observés dans les références (texture peau, couleurs, posture, environnement).
6. Toute la synthèse de texte et l'analyse d'image tournent en local via Ollama, sans appel à des LLM/VLM hébergés.
7. Temps total d'exécution < 8 minutes par espèce sur Mac M-series, après que les modèles soient téléchargés.
