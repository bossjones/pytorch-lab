.PHONY: setup lock sync env-works env-test test lint format typecheck check \
	clean jupyter ipython \
	setup-dataset-scratch-env download-dataset unzip-dataset zip-dataset \
	download-localization-dataset fetch-assets \
	install-postgres label-studio

# --- Environment (uv) -------------------------------------------------------
setup sync:
	uv sync

lock:
	uv lock

env-works:
	uv run python ./contrib/is-mps-available.py
	uv run python ./contrib/does-matplotlib-work.py

env-test: env-works

# --- Quality ----------------------------------------------------------------
test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyright

check: lint typecheck test

# --- Dev utilities ----------------------------------------------------------
clean:
	find . -name '*.pyc' -delete; \
	find . -name '*.pyo' -delete; \
	find . -name '__pycache__' -type d -exec rm -rf {} +; \
	rm -f .coverage

jupyter:
	uv run jupyter notebook

ipython:
	uv run ipython

# --- Dataset ----------------------------------------------------------------
setup-dataset-scratch-env:
	bash contrib/setup-dataset-scratch-env.sh

download-dataset: setup-dataset-scratch-env
	curl -L 'https://www.dropbox.com/s/8w1jkcvdzmh7khh/twitter_facebook_tiktok.zip?dl=1' > ./scratch/datasets/twitter_facebook_tiktok.zip
	unzip -l ./scratch/datasets/twitter_facebook_tiktok.zip

unzip-dataset:
	unzip ./scratch/datasets/twitter_facebook_tiktok.zip -d './scratch/datasets'
	rm -fv ./scratch/datasets/twitter_facebook_tiktok.zip

zip-dataset:
	bash contrib/zip-dataset.sh
	ls -ltah ./scratch/datasets/twitter_facebook_tiktok.zip

# Localization (screencropnet) assets. Provenance: ai_docs/screencropnet-assets.md
download-localization-dataset:
	uv run contrib/fetch_screencropnet_assets.py --dataset

fetch-assets:
	uv run contrib/fetch_screencropnet_assets.py --all

install-postgres:
	brew install postgresql@14

label-studio:
	label-studio
