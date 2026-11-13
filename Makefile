.RECIPEPREFIX = >
.PHONY: install test lint format data stream stream-down

install:
> python -m pip install --upgrade pip
> pip install -r requirements.txt
> pip install -e .

test:
> pytest

lint:
> ruff check .

format:
> ruff format .

data:
> bash scripts/download_skab.sh

stream:
> docker compose up --build

stream-down:
> docker compose down
