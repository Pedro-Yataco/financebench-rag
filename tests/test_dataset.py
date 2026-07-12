"""Tests for the FinanceBench dataset loader against a real 3-record fixture."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.dataset import (
    DocInfo,
    EvidenceItem,
    Question,
    load_doc_info,
    load_questions,
    referenced_doc_names,
    validate_doc_references,
)

FIXTURES = Path(__file__).parent / "fixtures"
QUESTIONS_FIXTURE = FIXTURES / "questions_sample.jsonl"
DOCS_FIXTURE = FIXTURES / "docs_sample.jsonl"


def _first_fixture_record() -> dict[str, Any]:
    line = QUESTIONS_FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    record: dict[str, Any] = json.loads(line)
    return record


def test_load_questions_parses_real_fixture() -> None:
    questions = load_questions(QUESTIONS_FIXTURE)

    assert len(questions) == 3
    assert {q.financebench_id for q in questions} == {
        "financebench_id_03029",
        "financebench_id_00499",
        "financebench_id_01865",
    }


def test_question_fields_and_optional_nulls() -> None:
    by_id = {q.financebench_id: q for q in load_questions(QUESTIONS_FIXTURE)}

    metrics = by_id["financebench_id_03029"]
    assert metrics.question_type == "metrics-generated"
    assert metrics.company == "3M"
    assert metrics.doc_name == "3M_2018_10K"
    assert metrics.answer == "$1577.00"
    assert metrics.justification is not None
    assert metrics.domain_question_num is None

    domain = by_id["financebench_id_00499"]
    assert domain.question_type == "domain-relevant"
    assert domain.domain_question_num == "dg06"

    novel = by_id["financebench_id_01865"]
    assert novel.question_type == "novel-generated"
    assert novel.question_reasoning is None
    assert novel.justification is None


def test_evidence_list_is_typed_and_bound_to_question_doc() -> None:
    by_id = {q.financebench_id: q for q in load_questions(QUESTIONS_FIXTURE)}

    multi = by_id["financebench_id_00499"]
    assert len(multi.evidence) == 3
    for item in multi.evidence:
        assert isinstance(item, EvidenceItem)
        assert item.doc_name == multi.doc_name
        assert isinstance(item.evidence_page_num, int)
        assert item.evidence_text
        assert item.evidence_text_full_page


def test_evidence_doc_name_mismatch_rejected() -> None:
    record = _first_fixture_record()
    record["evidence"][0]["doc_name"] = "SOME_OTHER_DOC"

    with pytest.raises(ValidationError, match="doc_name"):
        Question.model_validate(record)


def test_empty_evidence_rejected() -> None:
    record = _first_fixture_record()
    record["evidence"] = []

    with pytest.raises(ValidationError):
        Question.model_validate(record)


def test_unknown_question_type_rejected() -> None:
    record = _first_fixture_record()
    record["question_type"] = "made-up-type"

    with pytest.raises(ValidationError):
        Question.model_validate(record)


def test_malformed_record_rejected_by_loader(tmp_path: Path) -> None:
    record = _first_fixture_record()
    del record["question"]
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_questions(bad_file)


def test_non_integer_page_num_rejected() -> None:
    record = _first_fixture_record()
    record["evidence"][0]["evidence_page_num"] = "seven"

    with pytest.raises(ValidationError):
        Question.model_validate(record)


def test_load_doc_info_parses_real_fixture() -> None:
    infos = load_doc_info(DOCS_FIXTURE)

    assert set(infos) == {"3M_2018_10K", "3M_2022_10K"}
    info = infos["3M_2018_10K"]
    assert isinstance(info, DocInfo)
    assert info.company == "3M"
    assert info.doc_type == "10k"
    assert info.doc_period == 2018
    assert info.doc_link.startswith("http")


def test_load_doc_info_first_occurrence_wins_on_duplicate(tmp_path: Path) -> None:
    base = json.loads(DOCS_FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    first = dict(base, doc_period=2023)
    second = dict(base, doc_period=2022)
    dup_file = tmp_path / "dup.jsonl"
    dup_file.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")

    infos = load_doc_info(dup_file)

    assert len(infos) == 1
    assert infos[base["doc_name"]].doc_period == 2023


def test_referenced_doc_names_sorted_unique() -> None:
    questions = load_questions(QUESTIONS_FIXTURE)

    assert referenced_doc_names(questions) == ["3M_2018_10K", "3M_2022_10K"]


def test_validate_doc_references_passes_on_real_fixture() -> None:
    questions = load_questions(QUESTIONS_FIXTURE)
    infos = load_doc_info(DOCS_FIXTURE)

    validate_doc_references(questions, infos)


def test_validate_doc_references_raises_on_missing_doc() -> None:
    questions = load_questions(QUESTIONS_FIXTURE)

    with pytest.raises(ValueError, match="3M_2018_10K"):
        validate_doc_references(questions, {})
