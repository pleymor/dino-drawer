PY := .venv/bin/python
SPECIES ?= Tyrannosaurus rex
SLUG := $(shell echo "$(SPECIES)" | tr '[:upper:] ' '[:lower:]-')
OUT := out/$(SLUG)
ARGS ?=

.PHONY: help install test run run-batch papers images vision synthesis image video regen regen-all publish publish-all unpublish clean clean-cache

help:
	@echo "Targets:"
	@echo "  install        — uv venv + pip install -e .[dev]"
	@echo "  test           — pytest"
	@echo "  run            — full pipeline for SPECIES (default: 'Tyrannosaurus rex')"
	@echo "  run-batch      — run full pipeline in parallel for SPECIES_LIST='A|B|C' (pipe-separated)"
	@echo "  papers         — only fetch papers"
	@echo "  images         — only scrape reference images"
	@echo "  vision         — only VLM filtering"
	@echo "  synthesis      — only LLM synthesis"
	@echo "  image          — only diffusion (hero.png)"
	@echo "  video          — generate hero.mp4 from hero.png via Veo 3.1 (SPECIES=…)"
	@echo "  clean          — remove out/<slug>/ artifacts (keeps papers + raw refs)"
	@echo "  clean-cache    — also remove HF cache (~14 GB)"
	@echo ""
	@echo "  regen          — image + publish for SPECIES (after prompt tweak)"
	@echo "  regen-all      — regen every species under out/*/"
	@echo "  publish        — optimise + upload species to R2 (SPECIES=…)"
	@echo "  publish-all    — publish every species under out/*/"
	@echo "  unpublish      — remove species from R2 and catalog (SPECIES=…)"
	@echo ""
	@echo "Examples:"
	@echo "  make run SPECIES='Triceratops horridus'"
	@echo "  make run ARGS='--force-step synthesis'"
	@echo "  make run SPECIES='Smilodon fatalis' ARGS='--skip-refs --lang en'"
	@echo "  make regen SPECIES='Tyrannosaurus rex'"
	@echo "  make video SPECIES='Tyrannosaurus rex'"
	@echo "  make run-batch SPECIES_LIST='Allosaurus fragilis|Carnotaurus sastrei|Triceratops horridus'"

install:
	uv venv
	uv pip install -e ".[dev]"

test:
	$(PY) -m pytest

run:
	$(PY) -m dino_drawer "$(SPECIES)" $(ARGS)

run-batch:
	@if [ -z "$(SPECIES_LIST)" ]; then \
		echo "Usage: make run-batch SPECIES_LIST='Allosaurus fragilis|Carnotaurus sastrei|...'" >&2 ; \
		exit 1 ; \
	fi
	@mkdir -p out
	@echo "$(SPECIES_LIST)" | tr '|' '\n' | while IFS= read -r sp ; do \
		[ -z "$$sp" ] && continue ; \
		slug=$$(echo "$$sp" | tr '[:upper:] ' '[:lower:]-') ; \
		echo "==> launching '$$sp' (log: out/$$slug.log)" ; \
		$(PY) -m dino_drawer "$$sp" > "out/$$slug.log" 2>&1 & \
	done ; \
	wait
	@echo "==> all batch pipelines finished"

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

video:
	$(PY) -m dino_drawer.video "$(OUT)"

regen: image publish

regen-all:
	@for d in out/*/; do \
		slug=$$(basename $$d); \
		echo "==> Regenerating $$slug"; \
		$(PY) -m dino_drawer.image "out/$$slug" && \
		$(PY) -m dino_drawer.publish "out/$$slug" || exit 1; \
	done

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
	rm -f $(OUT)/hero.png $(OUT)/factsheet.json $(OUT)/refs.json $(OUT)/classifications_cache.json
	rm -rf $(OUT)/refs $(OUT)/_publish

clean-cache:
	rm -rf ~/.cache/huggingface
