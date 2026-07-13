"""Tests for the retrieval eval runner: plumbing, aggregation, results JSON."""

import json
import re
from pathlib import Path

import pytest

from src.dataset import Question, load_questions
from src.eval.runner import build_payload, evaluate_retrieval, write_results

FIXTURE = Path("tests/fixtures/questions_sample.jsonl")

# Fixture gold evidence pages are 0-based [59], [47, 49, 51], [24]; with the
# D-015 +1 offset the gold page_nos are {60}, {48, 50, 52}, {25}.
RANKINGS = {
    "3M_2018_10K": [60, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # gold first
    "3M_2022_10K": [1, 2, 48, 3, 4, 50, 52, 5, 6, 7],  # golds at ranks 3, 6, 7
}


class FakeRetriever:
    """Returns a canned ranking per doc and records every call."""

    def __init__(self, rankings: dict[str, list[int]]) -> None:
        self.rankings = rankings
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(self, question: str, doc_name: str, k: int) -> list[int]:
        self.calls.append((question, doc_name, k))
        return self.rankings.get(doc_name, list(range(100, 100 + k)))


def _questions() -> list[Question]:
    return load_questions(FIXTURE)


def test_evaluate_retrieval_computes_hand_checked_aggregates() -> None:
    questions = _questions()
    # q1 hits at rank 1; q2 at ranks 3/6/7; q3 (also 3M_2022_10K, gold {25}) misses
    rankings = dict(RANKINGS)
    retriever = FakeRetriever(rankings)

    result = evaluate_retrieval(questions, retriever, k=10)

    metrics = result["metrics"]
    assert metrics["recall@5"] == pytest.approx(2 / 3)
    assert metrics["recall@10"] == pytest.approx(2 / 3)
    assert metrics["full_recall@5"] == pytest.approx(1 / 3)
    assert metrics["full_recall@10"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)


def test_evaluate_retrieval_maps_gold_pages_with_offset() -> None:
    questions = _questions()
    retriever = FakeRetriever(RANKINGS)

    result = evaluate_retrieval(questions, retriever, k=10)

    gold_by_id = {r["financebench_id"]: r["gold_pages"] for r in result["per_question"]}
    assert gold_by_id["financebench_id_03029"] == [60]
    assert gold_by_id["financebench_id_00499"] == [48, 50, 52]
    assert gold_by_id["financebench_id_01865"] == [25]


def test_evaluate_retrieval_breaks_down_by_question_type() -> None:
    questions = _questions()
    retriever = FakeRetriever(RANKINGS)

    result = evaluate_retrieval(questions, retriever, k=10)

    by_type = result["by_question_type"]
    assert set(by_type) == {"metrics-generated", "domain-relevant", "novel-generated"}
    assert by_type["metrics-generated"]["n"] == 1
    assert by_type["metrics-generated"]["mrr"] == pytest.approx(1.0)
    assert by_type["domain-relevant"]["mrr"] == pytest.approx(1 / 3)
    assert by_type["novel-generated"]["recall@10"] == 0.0


def test_evaluate_retrieval_passes_question_doc_and_k_to_retriever() -> None:
    questions = _questions()
    retriever = FakeRetriever(RANKINGS)

    evaluate_retrieval(questions, retriever, k=7)

    assert retriever.calls == [(q.question, q.doc_name, 7) for q in questions]


def test_build_payload_carries_run_metadata() -> None:
    questions = _questions()
    evaluation = evaluate_retrieval(questions, FakeRetriever(RANKINGS), k=10)

    payload = build_payload(evaluation, retriever_name="bm25", k=10, smoke=True)

    assert payload["runner"] == "retrieval"
    assert payload["config"] == {"retriever": "bm25", "k": 10}
    assert payload["n_questions"] == 3
    assert payload["smoke"] is True
    assert re.fullmatch(r"[0-9a-f]{7,40}", payload["git_sha"])
    assert payload["timestamp"].endswith("+00:00")
    assert payload["metrics"] == evaluation["metrics"]
    assert len(payload["per_question"]) == 3


def test_write_results_creates_timestamped_json(tmp_path: Path) -> None:
    payload = {"runner": "retrieval", "metrics": {"recall@5": 0.5}}

    path = write_results(payload, results_dir=tmp_path)

    assert re.fullmatch(r"\d{8}T\d{6}Z_retrieval\.json", path.name)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


@pytest.mark.integration
def test_bm25_retrieves_pages_of_the_requested_doc() -> None:
    from src.ingest import parsed_path
    from src.pages import load_page_texts
    from src.retrieve import BM25PageRetriever

    questions = load_questions()
    parsed = [q for q in questions if parsed_path(q.doc_name).exists()]
    assert parsed, "no parsed docs on disk; run `make ingest` first"
    by_doc = {q.doc_name: q for q in parsed}
    doc_names = sorted(by_doc)[:2]

    retriever = BM25PageRetriever()
    for doc_name in doc_names:
        question = by_doc[doc_name]
        ranked = retriever.retrieve(question.question, doc_name, 10)
        assert 0 < len(ranked) <= 10
        assert set(ranked) <= set(load_page_texts(doc_name))
        assert all(isinstance(page, int) for page in ranked)
