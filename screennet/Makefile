.PHONY: train predict bad-predict worst-first tensorboard lint format typecheck test label-studio

train:
	uv run python main.py --to-disk --debug --interactive --epochs 10

predict:
	uv run python main.py --predict "~/Downloads/2020-11-25_10-47-32_867.jpeg" --weights ./models/ScreenNetV1.pth --interactive --to-disk

bad-predict:
	uv run python main.py --predict ../bad_image_test.jpg --weights ./models/ScreenNet_efficientnet_b0_basic_10_epochs.pth --interactive --to-disk

worst-first:
	uv run python main.py --to-disk --debug --predict ~/dev/bossjones/pytorch-lab/scratch/datasets/twitter_facebook_tiktok/test --weights ./models/ScreenNet_efficientnet_b0_basic_10_epochs.pth --worst-first --results ./worst-first.csv

tensorboard:
	uv run tensorboard --logdir runs/

# --- Quality (run from repo root config) ------------------------------------
lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyright main.py

test:
	uv run pytest

label-studio:
	label-studio
