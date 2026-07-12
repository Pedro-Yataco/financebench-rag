.PHONY: up down fetch-data ingest test test-integration lint typecheck

up:
	docker compose up -d

down:
	docker compose down

fetch-data:
	uv run python -m scripts.fetch_data

ingest:
	uv run python -m src.ingest

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src tests scripts
