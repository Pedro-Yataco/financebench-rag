.PHONY: fetch-data test test-integration lint typecheck

fetch-data:
	uv run python -m scripts.fetch_data

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src tests scripts
