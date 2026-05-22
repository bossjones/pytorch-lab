# screencropnet/Makefile — object localization (bbox regression) tasks.
# Run from the screencropnet/ directory (cwd is screencropnet/ for every recipe).
# `make` with no args prints help (see .DEFAULT_GOAL below).

# Absolute path of this directory (used by the label-studio docker mounts).
CURRENT_DIR := $(shell pwd)

# Committed sample screenshot, used in place of a machine-local ~/Downloads
# file so predict/autocrop work on a clean checkout and in CI. Path is relative
# to screencropnet/ (the fixture lives at repo-root tests/fixtures/).
FIXTURE_IMAGE := ../tests/fixtures/00006_twitter.PNG

# Bare `make` shows the help menu instead of launching a 40-epoch train run.
.DEFAULT_GOAL := help

.PHONY: train train_100 transfer_learning transfer_train train_one evaluate test \
	predict autocrop autocrop-resize bad-predict worst-first tensorboard \
	lint format typecheck get-best-model zip-dataset \
	label-studio label-studio-local label-studio-docker help

help: ## show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z0-9_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

train: ## train ObjLocModel for 40 epochs
	# --to-disk saves checkpoints/plots; --interactive shows progress; --debug verbose
	uv run python main.py --seed 42 --to-disk --debug --interactive --epochs 40

train_100: ## train ObjLocModel for 378 epochs (long run)
	# same as `train` but 378 epochs (the "100%" full-length schedule)
	uv run python main.py --seed 42 --to-disk --debug --interactive --epochs 378

transfer_learning: ## resume training (40 epochs) from the 40-epoch checkpoint
	# --weights loads an existing .pth to continue/fine-tune from
	uv run python main.py --seed 42 --to-disk --debug --interactive --epochs 40 --weights ./models/ScreenCropNetV1_ObjLocModelV1_pascalVOC_40_epochs.pth

transfer_train: transfer_learning ## alias for `transfer_learning`

train_one: ## train a single epoch from the 40-epoch checkpoint (smoke test)
	# 1-epoch run resuming from --weights; quick sanity check of the pipeline
	uv run python main.py --seed 42 --debug --epochs 1 --weights ./models/ScreenCropNetV1_ObjLocModelV1_pascalVOC_40_epochs.pth

evaluate: ## evaluate the 378-epoch model on the test set
	# --test switches to eval/test mode against the held-out split
	uv run python main.py --to-disk --debug --test --epochs 5 --weights ./models/ScreenCropNetV1_378_epochs.pth

test: evaluate ## alias for `evaluate`

predict: ## run inference on the committed fixture screenshot, save result to disk
	# --predict runs single-image inference; --to-disk writes the annotated output
	uv run python main.py --predict "$(FIXTURE_IMAGE)" --weights ./models/ScreenCropNetV1_378_epochs.pth --to-disk

autocrop: ## crop the fixture screenshot using the predicted bbox
	# --autocrop predicts a bbox then writes the cropped image to disk
	uv run python main.py --autocrop "$(FIXTURE_IMAGE)" --weights ./models/ScreenCropNetV1_378_epochs.pth --to-disk

autocrop-resize: ## autocrop the fixture screenshot with darkmode resize
	# --resize "darkmode" post-processes the crop for dark-mode screenshots
	uv run python main.py --resize "darkmode" --autocrop "$(FIXTURE_IMAGE)" --weights ./models/ScreenCropNetV1_378_epochs.pth --to-disk

worst-first: ## rank test-set images by highest loss, write worst-first.csv
	# --worst-first sorts predictions worst→best; --results writes the CSV report
	uv run python main.py --to-disk --debug --predict ~/dev/bossjones/pytorch-lab/scratch/datasets/twitter_facebook_tiktok/test --weights ./models/ScreenNet_efficientnet_b0_basic_10_epochs.pth --worst-first --results ./worst-first.csv

tensorboard: ## launch TensorBoard on the runs/ logdir
	# serves the training scalars/images logged under runs/
	uv run tensorboard --logdir runs/

# --- Quality (uses root pyproject.toml config) ------------------------------
lint: ## ruff lint the package
	# ruff check . (config in root pyproject.toml)
	uv run ruff check .

format: ## ruff-format the package
	# ruff format . (replaces black + isort)
	uv run ruff format .

typecheck: ## pyright type-check main.py
	# pyright in basic mode (config in root pyproject.toml)
	uv run pyright main.py

# --- Assets -----------------------------------------------------------------
get-best-model: ## download the pretrained 378-epoch .pth from Dropbox
	# create models/ then fetch the checkpoint into it
	mkdir -p models || true
	curl -L 'https://www.dropbox.com/s/9903r4jy02rmuzh/ScreenCropNetV1_378_epochs.pth?dl=1' > models/ScreenCropNetV1_378_epochs.pth

zip-dataset: ## zip the localization dataset and list the resulting archive
	# zip-dataset.sh builds the archive under ~/Downloads/datasets/
	bash ./zip-dataset.sh
	ls -ltah ~/Downloads/datasets/twitter_screenshots_localization_dataset.zip

# Label Studio runs as an isolated uv tool (`uv tool install label-studio`):
# its pinned requests/pillow versions conflict with the project venv, so it
# cannot live there. `uvx` == `uv tool run`.
label-studio: ## launch Label Studio annotation UI on http://localhost:8080
	uvx label-studio

label-studio-local: ## launch Label Studio serving ../scratch/datasets/ as local files
	# LOCAL_FILES_* lets you import the localization screenshots without uploading
	LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
	LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(CURRENT_DIR)/../scratch/datasets \
	uvx label-studio

label-studio-docker: ## run Label Studio in Docker (annotation UI on :8080)
	# mounts ./mydata (DB) and ./myfiles (local files) into the container
	docker run -it --rm -p 8080:8080 \
	-v $(CURRENT_DIR)/mydata:/label-studio/data \
	--env LOCAL_FILES_SERVING_ENABLED=true \
	--env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
	--env LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
	--env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
	--env JSON_LOG=1 \
	--env LOG_LEVEL=DEBUG \
	-v $(CURRENT_DIR)/myfiles:/label-studio/files \
	heartexlabs/label-studio:latest \
	label-studio
