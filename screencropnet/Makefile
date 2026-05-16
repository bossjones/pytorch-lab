CURRENT_DIR := $(shell pwd)

.PHONY: train train_100 transfer_learning transfer_train train_one evaluate test \
	predict autocrop autocrop-resize bad-predict worst-first tensorboard \
	lint format typecheck get-best-model zip-dataset label-studio

train:
	uv run python main.py --seed 42 --to-disk --debug --interactive --epochs 40

train_100:
	uv run python main.py --seed 42 --to-disk --debug --interactive --epochs 378

transfer_learning:
	uv run python main.py --seed 42 --to-disk --debug --interactive --epochs 40 --weights ./models/ScreenCropNetV1_ObjLocModelV1_pascalVOC_40_epochs.pth

transfer_train: transfer_learning

train_one:
	uv run python main.py --seed 42 --debug --epochs 1 --weights ./models/ScreenCropNetV1_ObjLocModelV1_pascalVOC_40_epochs.pth

evaluate:
	uv run python main.py --to-disk --debug --test --epochs 5 --weights ./models/ScreenCropNetV1_378_epochs.pth

test: evaluate

predict:
	uv run python main.py --predict "~/Downloads/IMG_6400.PNG" --weights ./models/ScreenCropNetV1_378_epochs.pth --to-disk

autocrop:
	uv run python main.py --autocrop "~/Downloads/2021-10-20_12-44-46_000.png" --weights ./models/ScreenCropNetV1_378_epochs.pth --to-disk

autocrop-resize:
	uv run python main.py --resize "darkmode" --autocrop "~/Downloads/2021-10-20_12-44-46_000.png" --weights ./models/ScreenCropNetV1_378_epochs.pth --to-disk

worst-first:
	uv run python main.py --to-disk --debug --predict ~/dev/bossjones/pytorch-lab/scratch/datasets/twitter_facebook_tiktok/test --weights ./models/ScreenNet_efficientnet_b0_basic_10_epochs.pth --worst-first --results ./worst-first.csv

tensorboard:
	uv run tensorboard --logdir runs/

# --- Quality (uses root pyproject.toml config) ------------------------------
lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyright main.py

# --- Assets -----------------------------------------------------------------
get-best-model:
	mkdir -p models || true
	curl -L 'https://www.dropbox.com/s/9903r4jy02rmuzh/ScreenCropNetV1_378_epochs.pth?dl=1' > models/ScreenCropNetV1_378_epochs.pth

zip-dataset:
	bash ./zip-dataset.sh
	ls -ltah ~/Downloads/datasets/twitter_screenshots_localization_dataset.zip

label-studio:
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
