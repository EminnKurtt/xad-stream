.RECIPEPREFIX = >
.PHONY: install test lint format data

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
