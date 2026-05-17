.PHONY: setup lock sync env-works env-test test test-cov test-cov-html open-cov lint format typecheck check \
	clean jupyter ipython \
	data-doctor data-setup \
	setup-dataset-scratch-env download-dataset unzip-dataset zip-dataset \
	download-localization-dataset fetch-assets \
	install-postgres label-studio \
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

# --- Dataset ----------------------------------------------------------------
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
	curl -L 'https://www.dropbox.com/s/8w1jkcvdzmh7khh/twitter_facebook_tiktok.zip?dl=1' > ./scratch/datasets/twitter_facebook_tiktok.zip
	unzip -l ./scratch/datasets/twitter_facebook_tiktok.zip

unzip-dataset: ## unzip the classification dataset into scratch/datasets/
	unzip ./scratch/datasets/twitter_facebook_tiktok.zip -d './scratch/datasets'
	rm -fv ./scratch/datasets/twitter_facebook_tiktok.zip

zip-dataset: ## re-zip the classification dataset
	bash contrib/zip-dataset.sh
	ls -ltah ./scratch/datasets/twitter_facebook_tiktok.zip

# Localization (screencropnet) assets. Provenance: ai_docs/screencropnet-assets.md
download-localization-dataset: ## download screencropnet localization dataset only
	uv run contrib/fetch_screencropnet_assets.py --dataset

fetch-assets: ## download screencropnet dataset, checkpoints, and sample image
	uv run contrib/fetch_screencropnet_assets.py --all

install-postgres: ## install PostgreSQL 14 via Homebrew
	brew install postgresql@14

label-studio: ## launch Label Studio annotation tool
	label-studio
