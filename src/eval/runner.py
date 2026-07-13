"""Retrieval evaluation runner.

Iterates questions through a page retriever (always scoped to the question's
gold doc, D-001), scores the ranked pages against the gold evidence pages
(0-based annotations mapped +1 per D-015), and writes one results JSON per
run to eval/results/. Runs with --limit are real-data subsets and are marked
"smoke": true — their numbers are never citable (D-011).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.dataset import Question, load_questions
from src.eval.metrics import full_recall_at_k, recall_at_k, reciprocal_rank
from src.pages import evidence_page_no
from src.retrieve import BM25PageRetriever, PageRetriever

RESULTS_DIR = Path("eval/results")
RECALL_KS = (5, 10)

# aggregate metric name -> per-question record key (means over questions)
_AGGREGATES = (
    {f"recall@{k}": f"recall@{k}" for k in RECALL_KS}
    | {f"full_recall@{k}": f"full_recall@{k}" for k in RECALL_KS}
    | {"mrr": "reciprocal_rank"}
)


def gold_page_nos(question: Question) -> list[int]:
    """Sorted unique gold page_nos for a question (D-015 mapping applied)."""
    return sorted({evidence_page_no(item.evidence_page_num) for item in question.evidence})


def _mean_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        aggregate: sum(record[key] for record in records) / len(records)
        for aggregate, key in _AGGREGATES.items()
    }


def evaluate_retrieval(
    questions: Sequence[Question], retriever: PageRetriever, k: int
) -> dict[str, Any]:
    """Score every question and aggregate overall and per question_type."""
    per_question = []
    for question in questions:
        gold = gold_page_nos(question)
        ranked = retriever.retrieve(question.question, question.doc_name, k)
        record: dict[str, Any] = {
            "financebench_id": question.financebench_id,
            "doc_name": question.doc_name,
            "question_type": question.question_type,
            "gold_pages": gold,
            "ranked_pages": ranked,
            "reciprocal_rank": reciprocal_rank(ranked, gold),
        }
        for recall_k in RECALL_KS:
            record[f"recall@{recall_k}"] = recall_at_k(ranked, gold, recall_k)
            record[f"full_recall@{recall_k}"] = full_recall_at_k(ranked, gold, recall_k)
        per_question.append(record)

    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in per_question:
        by_type.setdefault(record["question_type"], []).append(record)
    return {
        "metrics": _mean_metrics(per_question),
        "by_question_type": {
            question_type: {"n": len(records)} | _mean_metrics(records)
            for question_type, records in sorted(by_type.items())
        },
        "per_question": per_question,
    }


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def build_payload(
    evaluation: dict[str, Any], retriever_name: str, k: int, smoke: bool
) -> dict[str, Any]:
    """Full results-JSON payload: run metadata plus the evaluation output."""
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "runner": "retrieval",
        "config": {"retriever": retriever_name, "k": k},
        "n_questions": len(evaluation["per_question"]),
        "smoke": smoke,
        **evaluation,
    }


def write_results(payload: dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    """Write one timestamped JSON per run; returns the file path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{timestamp}_{payload['runner']}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate page retrieval over FinanceBench.")
    parser.add_argument("--retriever", choices=["bm25"], default="bm25")
    parser.add_argument("--k", type=int, default=10, help="pages retrieved per question")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="smoke run on the first N questions"
    )
    args = parser.parse_args(argv)

    questions = load_questions()
    if args.limit is not None:
        questions = questions[: args.limit]
    retriever = BM25PageRetriever()

    evaluation = evaluate_retrieval(questions, retriever, k=args.k)
    payload = build_payload(evaluation, args.retriever, args.k, smoke=args.limit is not None)
    path = write_results(payload)

    smoke_note = " (smoke - not citable)" if payload["smoke"] else ""
    print(f"{payload['n_questions']} questions, retriever={args.retriever}{smoke_note}")
    print("  ".join(f"{name}={value:.3f}" for name, value in payload["metrics"].items()))
    print(f"results written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
