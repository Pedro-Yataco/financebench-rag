.PHONY: up down fetch-data ingest chunk eval test test-integration lint typecheck

RETRIEVER ?= bm25

up:
	docker compose up -d

down:
	docker compose down

fetch-data:
	uv run python -m scripts.fetch_data

ingest:
	uv run python -m src.ingest

chunk:
	uv run python -m src.chunk

eval:
	uv run python -m src.eval.runner --retriever $(RETRIEVER) $(if $(LIMIT),--limit $(LIMIT),)

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src tests scripts
