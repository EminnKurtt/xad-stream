.RECIPEPREFIX = >
.PHONY: install test lint format

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
