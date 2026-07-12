"""Page-level retrieval metrics over ranked page lists.

Gold evidence for a question is a non-empty set of page numbers within its
gold document. Each function scores ONE question; the eval runner averages
the scores across questions (the mean of reciprocal_rank is the reported
MRR). Chunk retrievers are scored by first converting their chunk ranking
into a page ranking with pages_from_chunk_ranking.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence


def _check_gold(gold_pages: Collection[int]) -> None:
    if not gold_pages:
        raise ValueError("gold_pages must not be empty")


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_at_k(ranked_pages: Sequence[int], gold_pages: Collection[int], k: int) -> float:
    """1.0 if ANY gold page appears in the top-k of the ranking, else 0.0."""
    _check_gold(gold_pages)
    _check_k(k)
    gold = set(gold_pages)
    return float(any(page in gold for page in ranked_pages[:k]))


def full_recall_at_k(ranked_pages: Sequence[int], gold_pages: Collection[int], k: int) -> float:
    """1.0 if ALL gold pages appear in the top-k of the ranking, else 0.0."""
    _check_gold(gold_pages)
    _check_k(k)
    return float(set(gold_pages) <= set(ranked_pages[:k]))


def reciprocal_rank(ranked_pages: Sequence[int], gold_pages: Collection[int]) -> float:
    """1 / rank (1-based) of the first gold page in the ranking; 0.0 if none is ranked."""
    _check_gold(gold_pages)
    gold = set(gold_pages)
    for rank, page in enumerate(ranked_pages, start=1):
        if page in gold:
            return 1.0 / rank
    return 0.0


def pages_from_chunk_ranking(chunk_pages: Iterable[Sequence[int]]) -> list[int]:
    """Convert a ranked chunk list into a ranked page list by first-occurrence dedup.

    Chunks are visited in rank order and each chunk's pages in stored order,
    so pages tied at the same chunk rank keep a deterministic order.
    """
    seen: set[int] = set()
    ranking: list[int] = []
    for pages in chunk_pages:
        for page in pages:
            if page not in seen:
                seen.add(page)
                ranking.append(page)
    return ranking
