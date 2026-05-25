.PHONY: setup lock sync env-works env-test test test-cov test-cov-html open-cov lint format typecheck check \
	docstring-audit \
	clean jupyter ipython \
	data-doctor data-setup \
	setup-dataset-scratch-env download-dataset unzip-dataset zip-dataset \
	download-localization-dataset fetch-assets \
	install-postgres label-studio label-studio-local \
	claude-telegram-channel \
	help

.DEFAULT_GOAL := help

help: ## show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

# --- Environment (uv) -------------------------------------------------------
setup sync: ## create or update .venv from uv.lock
	uv sync

lock: ## re-resolve dependencies and update uv.lock
	uv lock

env-works: ## verify MPS and matplotlib are functional
	uv run python ./contrib/is-mps-available.py
	uv run python ./contrib/does-matplotlib-work.py

env-test: env-works ## alias for env-works

# --- Quality ----------------------------------------------------------------
test: ## run pytest
	uv run pytest

test-cov: ## run pytest with terminal coverage report
	uv run pytest --cov --cov-report=term-missing

test-cov-html: ## run pytest and generate htmlcov/ HTML coverage report
	uv run pytest --cov --cov-report=term-missing --cov-report=html

open-cov: ## open htmlcov/index.html in the default browser
	open htmlcov/index.html

lint: ## lint with ruff
	uv run ruff check .

format: ## format with ruff
	uv run ruff format .

typecheck: ## type-check with pyright
	uv run pyright

check: lint typecheck test ## run lint, typecheck, and tests

docstring-audit: ## regenerate docs/coverage_docs/report.md (docstring coverage)
	uv run contrib/docstring_coverage.py

# --- Dev utilities ----------------------------------------------------------
clean: ## remove build artifacts, caches, and coverage data
	find . -name '*.pyc' -delete; \
	find . -name '*.pyo' -delete; \
	find . -name '__pycache__' -type d -exec rm -rf {} +; \
	rm -f .coverage; \
	rm -rf htmlcov

jupyter: ## start Jupyter notebook server
	uv run jupyter notebook

ipython: ## launch IPython REPL
	uv run ipython

claude-telegram-channel: ## run Claude with official Telegram channel plugin (auto mode)
	claude --channels plugin:telegram@claude-plugins-official --enable-auto-mode

# --- Dataset ----------------------------------------------------------------
# FORCE is opt-in: unset by default. `make data-setup FORCE=1` (or
# `FORCE=1 make data-setup`) overrides every idempotency guard below and
# re-fetches classification + localization assets. Command-line/env vars
# propagate through the $(MAKE) sub-invocations in data-setup automatically.
FORCE ?=
DATASET_DIR := ./scratch/datasets/twitter_facebook_tiktok
DATASET_ZIP := ./scratch/datasets/twitter_facebook_tiktok.zip
DATASET_URL := https://www.dropbox.com/s/8w1jkcvdzmh7khh/twitter_facebook_tiktok.zip?dl=1

data-doctor: ## verify status of all expected dataset files and directories
	uv run contrib/data_doctor.py

data-setup: ## fetch all datasets end to end (classification + localization)
	$(MAKE) setup-dataset-scratch-env
	$(MAKE) download-dataset
	$(MAKE) unzip-dataset
	$(MAKE) fetch-assets

setup-dataset-scratch-env: ## create scratch/datasets/ directory layout
	bash contrib/setup-dataset-scratch-env.sh

download-dataset: setup-dataset-scratch-env ## download twitter/facebook/tiktok classification dataset
	@if [ -n "$(FORCE)" ]; then \
		echo "[force] re-downloading dataset zip"; \
		curl -L '$(DATASET_URL)' > $(DATASET_ZIP); \
	elif [ -d "$(DATASET_DIR)" ]; then \
		echo "[skip]  dataset already extracted at $(DATASET_DIR) (FORCE=1 to re-download)"; \
	elif [ -f "$(DATASET_ZIP)" ]; then \
		echo "[skip]  zip already present at $(DATASET_ZIP) (FORCE=1 to re-download)"; \
	else \
		curl -L '$(DATASET_URL)' > $(DATASET_ZIP); \
	fi
# unzip -l ./scratch/datasets/twitter_facebook_tiktok.zip

unzip-dataset: ## unzip the classification dataset into scratch/datasets/
	@if [ -d "$(DATASET_DIR)" ] && [ -z "$(FORCE)" ]; then \
		echo "[skip]  dataset already extracted at $(DATASET_DIR) (FORCE=1 to re-extract)"; \
	else \
		if [ -n "$(FORCE)" ]; then \
			echo "[force] removing $(DATASET_DIR) before re-extract"; \
			rm -rf "$(DATASET_DIR)"; \
		fi; \
		unzip -o $(DATASET_ZIP) -d ./scratch/datasets; \
		echo "[keep]  zip retained at $(DATASET_ZIP) (run 'make clean-dataset-zip' to remove)"; \
	fi

clean-dataset-zip: ## remove the downloaded classification zip
	rm -fv $(DATASET_ZIP)

zip-dataset: ## re-zip the classification dataset
	bash contrib/zip-dataset.sh
	ls -ltah ./scratch/datasets/twitter_facebook_tiktok.zip

# Localization (screencropnet) assets. Provenance: ai_docs/screencropnet-assets.md
download-localization-dataset: ## download screencropnet localization dataset only
	uv run contrib/fetch_screencropnet_assets.py --dataset $(if $(FORCE),--force,)

fetch-assets: ## download screencropnet dataset, checkpoints, and sample image
	uv run contrib/fetch_screencropnet_assets.py --all $(if $(FORCE),--force,)

install-postgres: ## install PostgreSQL 14 via Homebrew
	brew install postgresql@14

# Label Studio is installed as an isolated uv tool (`uv tool install
# label-studio`): its pinned requests/pillow versions conflict with this
# project's deps, so it cannot live in the project venv. `uvx` == `uv tool run`.
label-studio: ## launch Label Studio annotation UI on http://localhost:8080
	uvx label-studio

label-studio-local: ## launch Label Studio serving scratch/datasets/ as local files
	# LOCAL_FILES_* lets you import the on-disk screenshots without uploading them
	LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
	LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(CURDIR)/scratch/datasets \
	uvx label-studio
