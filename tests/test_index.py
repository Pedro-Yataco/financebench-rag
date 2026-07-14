"""Tests for src.index: point assembly, collection setup, idempotent upserts.

Unit tests use fake embedders and a recording fake client — no Qdrant, no
model downloads. The integration test indexes one real doc into a throwaway
collection and queries it back (requires `make up` plus model downloads).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from qdrant_client import models

from src.chunk import Chunk
from src.index import (
    COLLECTION,
    DENSE_DIM,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    build_points,
    ensure_collection,
    existing_point_ids,
    index_chunks,
    pending_chunks,
    point_id,
)


def make_chunk(i: int, doc_name: str = "TEST_2020_10K") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_name}:{i:04d}",
        doc_name=doc_name,
        company="TestCo",
        year=2020,
        doc_type="10k",
        pages=[i + 1],
        text=f"chunk number {i}",
    )


def fake_dense(texts: Sequence[str]) -> list[list[float]]:
    return [[float(len(text))] * DENSE_DIM for text in texts]


def fake_sparse(texts: Sequence[str]) -> list[models.SparseVector]:
    return [
        models.SparseVector(indices=[i], values=[float(len(text))]) for i, text in enumerate(texts)
    ]


@dataclass
class StoredPoint:
    id: str


@dataclass
class FakeClient:
    """Records the calls index.py makes against qdrant-client."""

    existing_collections: set[str] = field(default_factory=set)
    stored_ids: list[str] = field(default_factory=list)
    created: list[dict[str, Any]] = field(default_factory=list)
    payload_indexes: list[tuple[str, str]] = field(default_factory=list)
    upserts: list[tuple[str, list[models.PointStruct], bool]] = field(default_factory=list)

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.existing_collections

    def create_collection(self, collection_name: str, **kwargs: Any) -> bool:
        self.created.append({"collection_name": collection_name, **kwargs})
        self.existing_collections.add(collection_name)
        return True

    def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: Any, **kwargs: Any
    ) -> Any:
        self.payload_indexes.append((collection_name, field_name))
        return None

    def upsert(self, collection_name: str, points: Any, wait: bool = False, **kwargs: Any) -> Any:
        self.upserts.append((collection_name, list(points), wait))
        return None

    def scroll(
        self, collection_name: str, limit: int = 10, offset: Any = None, **kwargs: Any
    ) -> tuple[list[StoredPoint], Any]:
        start = 0 if offset is None else int(offset)
        page = [StoredPoint(id=i) for i in self.stored_ids[start : start + limit]]
        next_offset = start + limit if start + limit < len(self.stored_ids) else None
        return page, next_offset


class TestPointId:
    def test_is_a_stable_uuid(self) -> None:
        first = point_id("3M_2018_10K:0000")
        assert first == point_id("3M_2018_10K:0000")
        assert uuid.UUID(first)  # parses

    def test_distinct_chunk_ids_get_distinct_points(self) -> None:
        assert point_id("A:0000") != point_id("A:0001") != point_id("B:0000")


class TestBuildPoints:
    def test_assembles_named_vectors_and_full_payload(self) -> None:
        chunks = [make_chunk(0), make_chunk(1)]
        dense = fake_dense([c.text for c in chunks])
        sparse = fake_sparse([c.text for c in chunks])

        points = build_points(chunks, dense, sparse)

        assert [p.id for p in points] == [point_id(c.chunk_id) for c in chunks]
        first_vector = points[0].vector
        assert isinstance(first_vector, dict)
        assert first_vector[DENSE_VECTOR] == dense[0]
        assert first_vector[SPARSE_VECTOR] == sparse[0]
        assert points[0].payload == chunks[0].model_dump()

    def test_length_mismatch_raises(self) -> None:
        chunks = [make_chunk(0)]
        with pytest.raises(ValueError, match="length"):
            build_points(chunks, fake_dense(["a", "b"]), fake_sparse(["a"]))
        with pytest.raises(ValueError, match="length"):
            build_points(chunks, fake_dense(["a"]), fake_sparse(["a", "b"]))


class TestEnsureCollection:
    def test_creates_collection_and_doc_name_index_once(self) -> None:
        client = FakeClient()

        ensure_collection(client)

        assert len(client.created) == 1
        created = client.created[0]
        assert created["collection_name"] == COLLECTION
        assert created["vectors_config"][DENSE_VECTOR].size == DENSE_DIM
        assert created["vectors_config"][DENSE_VECTOR].distance == models.Distance.COSINE
        sparse_config = created["sparse_vectors_config"][SPARSE_VECTOR]
        assert sparse_config.modifier == models.Modifier.IDF
        assert client.payload_indexes == [(COLLECTION, "doc_name")]

    def test_existing_collection_left_untouched(self) -> None:
        client = FakeClient(existing_collections={COLLECTION})

        ensure_collection(client)

        assert client.created == []
        assert client.payload_indexes == []


class TestIndexChunks:
    def test_upserts_every_chunk_in_batches(self) -> None:
        client = FakeClient(existing_collections={COLLECTION})
        chunks = [make_chunk(i) for i in range(5)]

        total = index_chunks(
            chunks, client, fake_dense, fake_sparse, batch_size=2, log=lambda _: None
        )

        assert total == 5
        assert [len(points) for _, points, _ in client.upserts] == [2, 2, 1]
        assert all(name == COLLECTION for name, _, _ in client.upserts)
        assert all(wait for _, _, wait in client.upserts)
        upserted_ids = [p.id for _, points, _ in client.upserts for p in points]
        assert upserted_ids == [point_id(c.chunk_id) for c in chunks]

    def test_batches_pair_vectors_with_their_own_chunks(self) -> None:
        client = FakeClient(existing_collections={COLLECTION})
        chunks = [make_chunk(i) for i in range(3)]

        index_chunks(chunks, client, fake_dense, fake_sparse, batch_size=2, log=lambda _: None)

        for _, points, _ in client.upserts:
            for point in points:
                assert isinstance(point.vector, dict)
                assert point.payload is not None
                dense_vec = point.vector[DENSE_VECTOR]
                assert isinstance(dense_vec, list)
                # fake embedders encode len(text); mixups across batches would show
                assert dense_vec[0] == float(len(point.payload["text"]))


class TestResume:
    def test_existing_point_ids_paginates_the_scroll(self) -> None:
        ids = [point_id(f"A:{i:04d}") for i in range(5)]
        client = FakeClient(existing_collections={COLLECTION}, stored_ids=ids)

        assert existing_point_ids(client) == set(ids)

    def test_pending_chunks_keeps_only_unindexed_ones(self) -> None:
        chunks = [make_chunk(i) for i in range(4)]
        existing = {point_id(chunks[0].chunk_id), point_id(chunks[2].chunk_id)}

        assert pending_chunks(chunks, existing) == [chunks[1], chunks[3]]

    def test_pending_chunks_with_nothing_indexed_returns_all(self) -> None:
        chunks = [make_chunk(i) for i in range(2)]
        assert pending_chunks(chunks, set()) == chunks


@pytest.mark.integration
class TestQdrantIndexingIntegration:
    TEST_COLLECTION = "financebench_test_index"

    def test_index_one_doc_and_query_it_back(self) -> None:
        from qdrant_client import QdrantClient

        from src.chunk import CHUNKS_DIR, load_chunks
        from src.index import (
            QDRANT_URL,
            build_dense_embedder,
            build_sparse_doc_embedder,
            build_sparse_query_embedder,
        )

        files = sorted(CHUNKS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_size)
        if not files:
            pytest.skip("no chunk files under data/chunks")
        chunks = load_chunks(files[0])
        doc_name = chunks[0].doc_name

        client = QdrantClient(url=QDRANT_URL, timeout=30)
        if client.collection_exists(self.TEST_COLLECTION):
            client.delete_collection(self.TEST_COLLECTION)
        dense_embed = build_dense_embedder()
        sparse_embed = build_sparse_doc_embedder()
        try:
            ensure_collection(client, collection=self.TEST_COLLECTION)
            total = index_chunks(
                chunks,
                client,
                dense_embed,
                sparse_embed,
                collection=self.TEST_COLLECTION,
                log=lambda _: None,
            )
            assert total == len(chunks)
            assert client.count(self.TEST_COLLECTION, exact=True).count == len(chunks)

            question = chunks[0].text[:120]
            response = client.query_points(
                self.TEST_COLLECTION,
                prefetch=[
                    models.Prefetch(query=dense_embed([question])[0], using=DENSE_VECTOR, limit=5),
                    models.Prefetch(
                        query=build_sparse_query_embedder()([question])[0],
                        using=SPARSE_VECTOR,
                        limit=5,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=3,
                with_payload=True,
            )
            assert response.points
            top = response.points[0]
            assert top.payload is not None
            assert top.payload["doc_name"] == doc_name
            assert top.payload["pages"]
        finally:
            client.delete_collection(self.TEST_COLLECTION)
