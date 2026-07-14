"""Tests for the hybrid retriever: query shape, fusion output to pages.

Unit tests drive HybridChunkRetriever with fake embedders and a recording
fake client. The integration test queries the real Qdrant collection and
needs `make up` plus at least one indexed doc.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from qdrant_client import models

from src.index import DENSE_VECTOR, SPARSE_VECTOR
from src.retrieve import PREFETCH_LIMIT, HybridChunkRetriever

DENSE_QUERY = [0.5, 0.25]
SPARSE_QUERY = models.SparseVector(indices=[7], values=[1.0])


def fake_dense(texts: Sequence[str]) -> list[list[float]]:
    return [DENSE_QUERY for _ in texts]


def fake_sparse(texts: Sequence[str]) -> list[models.SparseVector]:
    return [SPARSE_QUERY for _ in texts]


@dataclass
class FakePoint:
    payload: dict[str, Any]


@dataclass
class FakeResponse:
    points: list[FakePoint]


@dataclass
class FakeSearchClient:
    """Returns canned fused points and records the query_points call."""

    pages_per_chunk: list[list[int]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def query_points(self, collection_name: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"collection_name": collection_name, **kwargs})
        return FakeResponse(points=[FakePoint(payload={"pages": p}) for p in self.pages_per_chunk])


def make_retriever(
    pages_per_chunk: list[list[int]], **kwargs: Any
) -> tuple[HybridChunkRetriever, FakeSearchClient]:
    client = FakeSearchClient(pages_per_chunk)
    return HybridChunkRetriever(client, fake_dense, fake_sparse, **kwargs), client


class TestHybridQueryShape:
    def test_both_prefetch_branches_filter_to_the_doc(self) -> None:
        retriever, client = make_retriever([[1]])

        retriever.retrieve("capex in 2018?", "3M_2018_10K", k=5)

        call = client.calls[0]
        prefetches = call["prefetch"]
        assert [p.using for p in prefetches] == [DENSE_VECTOR, SPARSE_VECTOR]
        expected_filter = models.Filter(
            must=[
                models.FieldCondition(key="doc_name", match=models.MatchValue(value="3M_2018_10K"))
            ]
        )
        assert all(p.filter == expected_filter for p in prefetches)
        assert all(p.limit == PREFETCH_LIMIT for p in prefetches)

    def test_sends_each_query_vector_to_its_own_branch(self) -> None:
        retriever, client = make_retriever([[1]])

        retriever.retrieve("q", "DOC", k=5)

        dense_prefetch, sparse_prefetch = client.calls[0]["prefetch"]
        assert dense_prefetch.query == DENSE_QUERY
        assert sparse_prefetch.query == SPARSE_QUERY

    def test_fuses_with_rrf_and_requests_pages_payload(self) -> None:
        retriever, client = make_retriever([[1]])

        retriever.retrieve("q", "DOC", k=5)

        call = client.calls[0]
        assert call["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
        assert call["limit"] == PREFETCH_LIMIT
        assert "pages" in call["with_payload"]

    def test_custom_collection_and_prefetch_limit(self) -> None:
        retriever, client = make_retriever([[1]], collection="other", prefetch_limit=7)

        retriever.retrieve("q", "DOC", k=3)

        call = client.calls[0]
        assert call["collection_name"] == "other"
        assert call["limit"] == 7
        assert all(p.limit == 7 for p in call["prefetch"])


class TestHybridPageRanking:
    def test_converts_chunk_ranking_to_deduped_pages(self) -> None:
        retriever, _ = make_retriever([[5], [5, 6], [2], [6]])

        assert retriever.retrieve("q", "DOC", k=10) == [5, 6, 2]

    def test_truncates_to_k_pages(self) -> None:
        retriever, _ = make_retriever([[5], [6], [2], [9]])

        assert retriever.retrieve("q", "DOC", k=2) == [5, 6]

    def test_no_hits_yield_no_pages(self) -> None:
        retriever, _ = make_retriever([])

        assert retriever.retrieve("q", "DOC", k=5) == []


@pytest.mark.integration
class TestHybridRetrievalIntegration:
    def test_returns_pages_of_the_requested_doc(self) -> None:
        from qdrant_client import QdrantClient

        from src.dataset import load_questions
        from src.index import (
            COLLECTION,
            QDRANT_URL,
            build_dense_embedder,
            build_sparse_query_embedder,
        )
        from src.pages import load_page_texts

        client = QdrantClient(url=QDRANT_URL, timeout=30)
        if not client.collection_exists(COLLECTION):
            pytest.skip(f"collection {COLLECTION} not present; run `make index` first")

        questions = load_questions()
        doc_filter = None
        question = None
        for candidate in questions:
            doc_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_name", match=models.MatchValue(value=candidate.doc_name)
                    )
                ]
            )
            if client.count(COLLECTION, count_filter=doc_filter, exact=True).count > 0:
                question = candidate
                break
        if question is None:
            pytest.skip("no question's doc is indexed yet")

        retriever = HybridChunkRetriever(
            client, build_dense_embedder(), build_sparse_query_embedder()
        )
        ranked = retriever.retrieve(question.question, question.doc_name, k=10)

        assert 0 < len(ranked) <= 10
        assert len(set(ranked)) == len(ranked)
        assert set(ranked) <= set(load_page_texts(question.doc_name))
