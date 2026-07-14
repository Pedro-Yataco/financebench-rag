# FinanceBench RAG

Retrieval-augmented generation over the [FinanceBench](https://github.com/patronus-ai/financebench)
open subset: 150 questions on SEC filings (10-K / 10-Q / 8-K / earnings reports), each
annotated with a gold answer and evidence pages.

Hand-rolled pipeline — no RAG frameworks: Docling parsing, hybrid retrieval (BM25 + dense
embeddings in Qdrant), cross-encoder reranking, and a typed generation step with abstention
support. Evaluated with page-level retrieval metrics and an end-to-end grading harness.

## Status

Work in progress — eval harness, BM25 baseline, and hybrid retrieval (dense BGE-M3 +
sparse BM25 over Qdrant) in place.

## Results

Retrieval is evaluated in the **oracle-document setting**: every query is filtered to the
question's gold document, so the numbers measure in-document page ranking, not cross-document
routing (and are not comparable to the FinanceBench paper's shared-vector-store setting).
Metrics are page-level over all 150 open-source questions: a hit means any gold evidence
page appears in the top-k; MRR uses the rank of the first gold page.

| Retriever | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 over page text (baseline) | 0.367 | 0.487 | 0.243 |
| Hybrid: dense (BGE-M3) + sparse BM25, RRF | 0.453 | 0.607 | 0.319 |

The hybrid retriever queries 512-token structure-aware chunks (Docling HybridChunker) with
dense and sparse prefetch fused by reciprocal rank fusion, then ranks pages by first
occurrence in the fused chunk list. The largest gain is on metrics-generated questions —
numeric questions whose wording rarely matches the filing text — where recall@5 goes from
0.14 (BM25) to 0.40; novel-generated reaches 0.62. Every row is produced by `make eval`,
which writes a full results JSON (config, git SHA, per-question records) to `eval/results/`.

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
make up            # start Qdrant (docker compose)
make chunk         # chunk parsed docs into data/chunks/ (resumable)
make index         # embed chunks and upsert into Qdrant (resumable)
make eval          # retrieval eval (RETRIEVER=bm25|hybrid, LIMIT=N for a smoke subset)
```
