"""Chunk embedding and Qdrant indexing.

Each chunk becomes one point with two named vectors: dense BGE-M3 via
sentence-transformers (fastembed 0.8.0 ships no BGE-M3 dense model, D-017)
and sparse BM25 term weights via fastembed, with IDF applied server-side
(Modifier.IDF on the sparse vector config). Point IDs are UUID5 hashes of
chunk_id, so re-running the indexer overwrites points instead of duplicating
them. The full chunk record rides along as payload; doc_name gets a keyword
index for the oracle-doc filter (D-001).
"""

from __future__ import annotations

import argparse
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from qdrant_client import QdrantClient, models

if TYPE_CHECKING:
    # runtime import deferred: src.chunk drags in docling's chunker (and with
    # it transformers), which retrieval-only importers of this module skip
    from src.chunk import Chunk

QDRANT_URL = "http://localhost:6333"
COLLECTION = "financebench"
DENSE_MODEL_ID = "BAAI/bge-m3"
SPARSE_MODEL_ID = "Qdrant/bm25"
DENSE_DIM = 1024
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
DENSE_BATCH_SIZE = 8  # CPU encoder batch, sized for 512-token sequences
UPSERT_BATCH_SIZE = 64

DenseEmbedder = Callable[[Sequence[str]], list[list[float]]]
SparseEmbedder = Callable[[Sequence[str]], list[models.SparseVector]]

# uuid5 namespace for chunk_id -> point id (any fixed UUID works; this one is
# uuid5(NAMESPACE_DNS, "financebench-rag"))
_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "financebench-rag")


class IndexClient(Protocol):
    """The slice of qdrant-client that indexing uses (fakeable in unit tests)."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(self, collection_name: str, **kwargs: Any) -> bool: ...

    def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: Any, **kwargs: Any
    ) -> Any: ...

    def upsert(
        self, collection_name: str, points: Any, wait: bool = False, **kwargs: Any
    ) -> Any: ...

    def scroll(self, collection_name: str, **kwargs: Any) -> tuple[list[Any], Any]: ...


def point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point ID for a chunk: same chunk, same point."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def build_points(
    chunks: Sequence[Chunk],
    dense: Sequence[list[float]],
    sparse: Sequence[models.SparseVector],
) -> list[models.PointStruct]:
    """Pair each chunk with its two vectors; the payload is the full record."""
    if not (len(chunks) == len(dense) == len(sparse)):
        raise ValueError(
            f"length mismatch: {len(chunks)} chunks, {len(dense)} dense, {len(sparse)} sparse"
        )
    return [
        models.PointStruct(
            id=point_id(chunk.chunk_id),
            vector={DENSE_VECTOR: dense_vec, SPARSE_VECTOR: sparse_vec},
            payload=chunk.model_dump(),
        )
        for chunk, dense_vec, sparse_vec in zip(chunks, dense, sparse, strict=True)
    ]


def ensure_collection(client: IndexClient, collection: str = COLLECTION) -> None:
    """Create the collection and its doc_name filter index if absent."""
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="doc_name",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def existing_point_ids(client: IndexClient, collection: str = COLLECTION) -> set[str]:
    """IDs already stored in the collection, via one scroll pass.

    With deterministic point IDs this makes indexing resumable: a rerun
    embeds only the chunks whose points are missing (the multi-hour CPU
    embed must survive interruptions).
    """
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=4096,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(point.id) for point in points)
        if offset is None:
            return ids


def pending_chunks(chunks: Sequence[Chunk], existing_ids: set[str]) -> list[Chunk]:
    """Chunks whose points are not yet in the collection."""
    return [chunk for chunk in chunks if point_id(chunk.chunk_id) not in existing_ids]


def _log_flushed(message: str) -> None:
    print(message, flush=True)


def index_chunks(
    chunks: Sequence[Chunk],
    client: IndexClient,
    dense_embed: DenseEmbedder,
    sparse_embed: SparseEmbedder,
    collection: str = COLLECTION,
    batch_size: int = UPSERT_BATCH_SIZE,
    log: Callable[[str], None] = _log_flushed,
) -> int:
    """Embed and upsert chunks in batches; returns the number indexed."""
    start = time.perf_counter()
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        texts = [chunk.text for chunk in batch]
        points = build_points(batch, dense_embed(texts), sparse_embed(texts))
        client.upsert(collection_name=collection, points=points, wait=True)
        done = offset + len(batch)
        rate = done / (time.perf_counter() - start)
        log(f"[{done}/{len(chunks)}] indexed ({rate:.1f} chunks/s)")
    return len(chunks)


def build_dense_embedder() -> DenseEmbedder:
    """BGE-M3 dense encoder (CPU); loads the model on first call site."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DENSE_MODEL_ID)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors = model.encode(
            list(texts),
            batch_size=DENSE_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    return embed


def _to_sparse_vectors(embeddings: Any) -> list[models.SparseVector]:
    return [
        models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in embeddings
    ]


def build_sparse_doc_embedder() -> SparseEmbedder:
    """BM25 document-side term weights (fastembed); IDF is applied by Qdrant."""
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding(SPARSE_MODEL_ID)

    def embed(texts: Sequence[str]) -> list[models.SparseVector]:
        return _to_sparse_vectors(model.embed(list(texts)))

    return embed


def build_sparse_query_embedder() -> SparseEmbedder:
    """BM25 query-side weights (term frequency only, per the BM25 formula)."""
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding(SPARSE_MODEL_ID)

    def embed(texts: Sequence[str]) -> list[models.SparseVector]:
        return _to_sparse_vectors(model.query_embed(list(texts)))

    return embed


def load_all_chunks(chunks_dir: Path | None = None, limit: int | None = None) -> list[Chunk]:
    """Chunks of every doc (sorted by file name); limit caps the DOC count."""
    from src.chunk import CHUNKS_DIR, load_chunks

    files = sorted((chunks_dir or CHUNKS_DIR).glob("*.jsonl"))
    if limit is not None:
        files = files[:limit]
    return [chunk for file in files for chunk in load_chunks(file)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed chunks and upsert them into Qdrant.")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="index only the first N docs"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="drop the collection first (stale-point safety)"
    )
    args = parser.parse_args(argv)

    chunks = load_all_chunks(limit=args.limit)
    if not chunks:
        print("no chunks found under data/chunks — run `make chunk` first")
        return 1

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    if args.recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"dropped collection {COLLECTION}")
    ensure_collection(client)

    todo = pending_chunks(chunks, existing_point_ids(client))
    print(
        f"{len(chunks)} chunks / {len(chunks) - len(todo)} already indexed / {len(todo)} to embed"
        f" (dense={DENSE_MODEL_ID}, sparse={SPARSE_MODEL_ID})",
        flush=True,
    )
    if todo:
        index_chunks(todo, client, build_dense_embedder(), build_sparse_doc_embedder())

    count = client.count(COLLECTION, exact=True).count
    print(f"collection {COLLECTION}: {count} points / {len(chunks)} chunks on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
