# FinanceBench RAG

Retrieval-augmented generation over the [FinanceBench](https://github.com/patronus-ai/financebench)
open subset: 150 questions on SEC filings (10-K / 10-Q / 8-K / earnings reports), each
annotated with a gold answer and evidence pages.

Hand-rolled pipeline — no RAG frameworks: Docling parsing, hybrid retrieval (BM25 + dense
embeddings in Qdrant), cross-encoder reranking, and a typed generation step with abstention
support. Evaluated with page-level retrieval metrics and an end-to-end grading harness.

## Status

Work in progress — project scaffold.

## Requirements

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- Docker (Qdrant)
- GNU Make

## Development

```sh
uv sync            # install pinned dependencies
make test          # unit test suite
make lint          # ruff check + format check
make typecheck     # mypy
```
