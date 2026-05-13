PY := .venv/bin/python
SPECIES ?= Tyrannosaurus rex
SLUG := $(shell echo "$(SPECIES)" | tr '[:upper:] ' '[:lower:]-')
OUT := out/$(SLUG)
ARGS ?=

.PHONY: help install test run papers images vision synthesis image compose publish publish-all unpublish clean clean-cache

help:
	@echo "Targets:"
	@echo "  install        — uv venv + pip install -e .[dev] + playwright chromium"
	@echo "  test           — pytest"
	@echo "  run            — full pipeline for SPECIES (default: 'Tyrannosaurus rex')"
	@echo "  papers         — only fetch papers"
	@echo "  images         — only scrape reference images"
	@echo "  vision         — only VLM filtering"
	@echo "  synthesis      — only LLM synthesis"
	@echo "  image          — only diffusion (hero + skull + silhouette)"
	@echo "  compose        — only final HTML→PNG render"
	@echo "  clean          — remove out/<slug>/ artifacts (keeps papers + raw refs)"
	@echo "  clean-cache    — also remove HF cache (~14 GB)"
	@echo ""
	@echo "  publish        — optimise + upload species to R2 (SPECIES=…)"
	@echo "  publish-all    — publish every species under out/*/"
	@echo "  unpublish      — remove species from R2 and catalog (SPECIES=…)"
	@echo ""
	@echo "Examples:"
	@echo "  make run SPECIES='Triceratops horridus'"
	@echo "  make run ARGS='--force-step synthesis'"
	@echo "  make run SPECIES='Smilodon fatalis' ARGS='--skip-refs --lang en'"

install:
	uv venv
	$(PY) -m pip install --upgrade pip
	uv pip install -e ".[dev]"
	$(PY) -m playwright install chromium

test:
	$(PY) -m pytest

run:
	$(PY) -m dino_drawer "$(SPECIES)" $(ARGS)

papers:
	$(PY) -m dino_drawer.research.papers "$(SPECIES)"

images:
	$(PY) -m dino_drawer.research.images "$(SPECIES)"

vision:
	$(PY) -m dino_drawer.vision "$(OUT)"

synthesis:
	$(PY) -m dino_drawer.synthesis "$(OUT)"

image:
	$(PY) -m dino_drawer.image "$(OUT)"

compose:
	$(PY) -m dino_drawer.compose "$(OUT)"

publish:
	$(PY) -m dino_drawer.publish "$(OUT)"

publish-all:
	@for d in out/*/; do \
		slug=$$(basename $$d); \
		echo "==> Publishing $$slug"; \
		$(PY) -m dino_drawer.publish "out/$$slug" || exit 1; \
	done

unpublish:
	$(PY) -m dino_drawer.publish "$(OUT)" --unpublish

clean:
	rm -f $(OUT)/hero.png $(OUT)/skull.png $(OUT)/silhouette.svg $(OUT)/final.png $(OUT)/_infographic.html $(OUT)/factsheet.json $(OUT)/refs.json $(OUT)/classifications_cache.json
	rm -rf $(OUT)/refs

clean-cache:
	rm -rf ~/.cache/huggingface
