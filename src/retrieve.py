"""Page retrieval over parsed documents.

Phase-1 baseline: classic BM25 (bm25s) over per-page plain text, one
in-memory index per document. Phase 2 adds the hybrid retriever: dense and
sparse prefetch over the Qdrant chunk collection fused with RRF, with the
fused chunk ranking converted to a page ranking (D-002). Retrieval is
always scoped to a single doc_name — the oracle-doc setting (D-001).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import bm25s
from qdrant_client import models

from src.eval.metrics import pages_from_chunk_ranking
from src.index import COLLECTION as CHUNK_COLLECTION
from src.index import DENSE_VECTOR, SPARSE_VECTOR, DenseEmbedder, SparseEmbedder
from src.ingest import PARSED_DIR
from src.pages import load_page_texts

PREFETCH_LIMIT = 20  # chunks fetched per branch and kept after fusion


class PageRetriever(Protocol):
    """Anything that ranks the pages of one document for a question."""

    def retrieve(self, question: str, doc_name: str, k: int) -> list[int]: ...


class SearchClient(Protocol):
    """The slice of qdrant-client that hybrid retrieval uses."""

    def query_points(self, collection_name: str, **kwargs: Any) -> Any: ...


class BM25PageRetriever:
    """BM25 over page text with one lazily built index per doc."""

    def __init__(self, parsed_dir: Path = PARSED_DIR) -> None:
        self._parsed_dir = parsed_dir
        self._indexes: dict[str, tuple[bm25s.BM25, list[int]]] = {}

    def _index(self, doc_name: str) -> tuple[bm25s.BM25, list[int]]:
        if doc_name not in self._indexes:
            pages = load_page_texts(doc_name, self._parsed_dir)
            page_nos = sorted(pages)
            index = bm25s.BM25()
            corpus = [pages[no] for no in page_nos]
            tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
            index.index(tokens, show_progress=False)
            self._indexes[doc_name] = (index, page_nos)
        return self._indexes[doc_name]

    def retrieve(self, question: str, doc_name: str, k: int) -> list[int]:
        """Top-k page_nos of doc_name ranked by BM25 score for the question."""
        index, page_nos = self._index(doc_name)
        query = bm25s.tokenize(question, stopwords="en", show_progress=False)
        results, _ = index.retrieve(query, k=min(k, len(page_nos)), show_progress=False)
        return [page_nos[int(i)] for i in results[0]]


class HybridChunkRetriever:
    """Dense + sparse prefetch over the chunk collection, fused with RRF.

    Both prefetch branches carry the doc_name filter (oracle setting, D-001);
    the fused top chunks become a page ranking by first-occurrence dedup of
    each chunk's pages (D-002), so fewer than k pages can come back when the
    top chunks concentrate on few pages.
    """

    def __init__(
        self,
        client: SearchClient,
        dense_embed: DenseEmbedder,
        sparse_embed: SparseEmbedder,
        collection: str = CHUNK_COLLECTION,
        prefetch_limit: int = PREFETCH_LIMIT,
    ) -> None:
        self._client = client
        self._dense_embed = dense_embed
        self._sparse_embed = sparse_embed
        self._collection = collection
        self.prefetch_limit = prefetch_limit

    def retrieve(self, question: str, doc_name: str, k: int) -> list[int]:
        """Top-k page_nos of doc_name from the RRF-fused chunk ranking."""
        doc_filter = models.Filter(
            must=[models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name))]
        )
        response = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=self._dense_embed([question])[0],
                    using=DENSE_VECTOR,
                    filter=doc_filter,
                    limit=self.prefetch_limit,
                ),
                models.Prefetch(
                    query=self._sparse_embed([question])[0],
                    using=SPARSE_VECTOR,
                    filter=doc_filter,
                    limit=self.prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self.prefetch_limit,
            with_payload=["pages"],
        )
        ranked_chunk_pages = [point.payload["pages"] for point in response.points]
        return pages_from_chunk_ranking(ranked_chunk_pages)[:k]
