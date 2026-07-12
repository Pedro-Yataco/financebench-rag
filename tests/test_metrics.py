"""Tests for page-level retrieval metrics: hand-computed cases per DECISIONS D-002."""

import pytest

from src.eval.metrics import (
    full_recall_at_k,
    pages_from_chunk_ranking,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_hits_when_gold_page_ranked_first() -> None:
    assert recall_at_k([5, 2, 3], {5}, k=5) == 1.0


def test_recall_at_k_hits_at_exactly_rank_k() -> None:
    assert recall_at_k([1, 2, 7], {7}, k=3) == 1.0


def test_recall_at_k_misses_when_gold_just_outside_top_k() -> None:
    assert recall_at_k([1, 2, 7], {7}, k=2) == 0.0


def test_recall_at_k_any_gold_page_counts() -> None:
    # multi-evidence question: ONE gold page in top-k is a hit (D-002)
    assert recall_at_k([9, 1], {4, 9}, k=1) == 1.0


def test_recall_at_k_misses_when_no_gold_ranked() -> None:
    assert recall_at_k([1, 2, 3], {4}, k=10) == 0.0


def test_recall_at_k_empty_ranking_is_zero() -> None:
    assert recall_at_k([], {4}, k=5) == 0.0


def test_full_recall_requires_all_gold_pages_in_top_k() -> None:
    assert full_recall_at_k([8, 5, 2], {2, 8}, k=3) == 1.0
    assert full_recall_at_k([8, 5, 2], {2, 8}, k=2) == 0.0


def test_full_recall_equals_recall_for_single_gold_page() -> None:
    assert full_recall_at_k([3], {3}, k=1) == 1.0
    assert recall_at_k([3], {3}, k=1) == 1.0


def test_full_recall_zero_when_a_gold_page_never_ranked() -> None:
    assert full_recall_at_k([8, 1, 4], {2, 8}, k=10) == 0.0
    # ... while any-page recall still counts it as a hit
    assert recall_at_k([8, 1, 4], {2, 8}, k=10) == 1.0


def test_reciprocal_rank_is_one_for_gold_at_first_position() -> None:
    assert reciprocal_rank([6, 1, 2], {6}) == 1.0


def test_reciprocal_rank_for_gold_at_third_position() -> None:
    assert reciprocal_rank([1, 2, 6], {6}) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_gold_page_found() -> None:
    assert reciprocal_rank([1, 9, 4], {4, 9}) == 0.5


def test_reciprocal_rank_zero_when_gold_absent() -> None:
    assert reciprocal_rank([1, 2, 3], {6}) == 0.0
    assert reciprocal_rank([], {6}) == 0.0


def test_pages_from_chunk_ranking_first_occurrence_dedup() -> None:
    assert pages_from_chunk_ranking([[3, 4], [4, 5], [1]]) == [3, 4, 5, 1]


def test_pages_from_chunk_ranking_preserves_within_chunk_order() -> None:
    # pages tied at the same chunk rank keep the chunk's stored page order
    assert pages_from_chunk_ranking([[7, 3], [3, 9]]) == [7, 3, 9]


def test_pages_from_chunk_ranking_dedups_within_a_single_chunk() -> None:
    assert pages_from_chunk_ranking([[2, 2, 3]]) == [2, 3]


def test_pages_from_chunk_ranking_handles_empty_inputs() -> None:
    assert pages_from_chunk_ranking([]) == []
    assert pages_from_chunk_ranking([[], [5]]) == [5]


def test_empty_gold_pages_rejected() -> None:
    with pytest.raises(ValueError, match="gold_pages"):
        recall_at_k([1], set(), k=5)
    with pytest.raises(ValueError, match="gold_pages"):
        full_recall_at_k([1], set(), k=5)
    with pytest.raises(ValueError, match="gold_pages"):
        reciprocal_rank([1], set())


def test_k_below_one_rejected() -> None:
    with pytest.raises(ValueError, match="k"):
        recall_at_k([1], {1}, k=0)
    with pytest.raises(ValueError, match="k"):
        full_recall_at_k([1], {1}, k=0)


def test_chunk_ranking_scored_end_to_end() -> None:
    # runner path: ranked chunks -> page ranking -> metrics, all hand-computed
    ranked_pages = pages_from_chunk_ranking([[10, 11], [2], [11, 12]])

    assert ranked_pages == [10, 11, 2, 12]
    assert recall_at_k(ranked_pages, {12}, k=3) == 0.0
    assert recall_at_k(ranked_pages, {12}, k=5) == 1.0
    assert full_recall_at_k(ranked_pages, {11, 12}, k=5) == 1.0
    assert reciprocal_rank(ranked_pages, {12}) == 0.25
