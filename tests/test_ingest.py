"""Tests for the ingest orchestration: resume/skip, failures, limit, atomic cache."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.dataset import pdf_path, referenced_doc_names
from src.ingest import IngestResult, format_summary, ingest_docs, parsed_path


def _ok_parse(pdf: Path) -> dict[str, Any]:
    return {"source": pdf.name}


def test_ingest_parses_uncached_docs_and_writes_json(tmp_path: Path) -> None:
    names = ["DOC_A", "DOC_B"]

    result = ingest_docs(names, _ok_parse, parsed_dir=tmp_path, log=lambda _: None)

    assert result.parsed == names
    assert result.skipped == []
    assert result.failures == []
    for name in names:
        cached = json.loads(parsed_path(name, tmp_path).read_text(encoding="utf-8"))
        assert cached == {"source": f"{name}.pdf"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_ingest_skips_cached_docs(tmp_path: Path) -> None:
    parsed_path("DOC_A", tmp_path).parent.mkdir(parents=True, exist_ok=True)
    parsed_path("DOC_A", tmp_path).write_text('{"cached": true}', encoding="utf-8")
    calls: list[str] = []
    logs: list[str] = []

    def parse(pdf: Path) -> dict[str, Any]:
        calls.append(pdf.name)
        return _ok_parse(pdf)

    result = ingest_docs(["DOC_A", "DOC_B"], parse, parsed_dir=tmp_path, log=logs.append)

    assert result.skipped == ["DOC_A"]
    assert result.parsed == ["DOC_B"]
    assert calls == ["DOC_B.pdf"]
    assert any("skipped (cached)" in line for line in logs)
    # the cached file is left untouched
    assert json.loads(parsed_path("DOC_A", tmp_path).read_text(encoding="utf-8")) == {
        "cached": True
    }


def test_ingest_records_failure_and_continues(tmp_path: Path) -> None:
    def parse(pdf: Path) -> dict[str, Any]:
        if "BAD" in pdf.name:
            raise ValueError("boom")
        return _ok_parse(pdf)

    result = ingest_docs(["BAD_DOC", "GOOD_DOC"], parse, parsed_dir=tmp_path, log=lambda _: None)

    assert result.failures == [("BAD_DOC", "ValueError: boom")]
    assert result.parsed == ["GOOD_DOC"]
    # a failed parse must leave no cache file (rerun retries it)
    assert not parsed_path("BAD_DOC", tmp_path).exists()
    assert parsed_path("GOOD_DOC", tmp_path).exists()


def test_ingest_limit_processes_only_first_n(tmp_path: Path) -> None:
    names = ["DOC_A", "DOC_B", "DOC_C", "DOC_D"]

    result = ingest_docs(names, _ok_parse, parsed_dir=tmp_path, limit=2, log=lambda _: None)

    assert result.parsed == ["DOC_A", "DOC_B"]
    assert not parsed_path("DOC_C", tmp_path).exists()
    assert not parsed_path("DOC_D", tmp_path).exists()


def test_ingest_limit_counts_cached_docs_within_window(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    for name in ["DOC_A", "DOC_B"]:
        parsed_path(name, tmp_path).write_text("{}", encoding="utf-8")

    result = ingest_docs(
        ["DOC_A", "DOC_B", "DOC_C"], _ok_parse, parsed_dir=tmp_path, limit=2, log=lambda _: None
    )

    assert result.skipped == ["DOC_A", "DOC_B"]
    assert result.parsed == []


def test_native_failures_extracts_page_failure_lines() -> None:
    from src.ingest import native_failures

    stderr_text = (
        "Loading weights: 100%|##########| 770/770\n"
        "Stage preprocess failed for run 4, pages [10]: std::bad_alloc\n"
        "Finished converting pages 112/112\n"
        "Stage preprocess failed for run 4, pages [11]: std::bad_alloc\n"
    )

    assert native_failures(stderr_text) == [
        "Stage preprocess failed for run 4, pages [10]: std::bad_alloc",
        "Stage preprocess failed for run 4, pages [11]: std::bad_alloc",
    ]
    assert native_failures("Loading weights: 100%\nall good\n") == []


def test_check_conversion_passes_on_clean_success() -> None:
    from src.ingest import check_conversion

    check_conversion("success", "Loading weights: 100%\n")


def test_check_conversion_raises_on_native_page_failures() -> None:
    from src.ingest import check_conversion

    with pytest.raises(RuntimeError, match=r"pages \[10\]"):
        check_conversion(
            "success", "Stage preprocess failed for run 4, pages [10]: std::bad_alloc\n"
        )


def test_check_conversion_raises_on_non_success_status() -> None:
    from src.ingest import check_conversion

    with pytest.raises(RuntimeError, match="partial_success"):
        check_conversion("partial_success", "")


def test_format_summary_reports_counts_and_failures() -> None:
    result = IngestResult(
        parsed=["DOC_C"],
        skipped=["DOC_A", "DOC_B"],
        failures=[("DOC_D", "ValueError: boom")],
    )

    summary = format_summary(result)

    assert "1 parsed / 2 skipped (cached) / 1 failed" in summary
    assert "DOC_D" in summary
    assert "ValueError: boom" in summary


@pytest.mark.integration
def test_ingest_one_real_pdf_end_to_end(tmp_path: Path) -> None:
    from src.dataset import load_questions
    from src.ingest import build_docling_parser

    names = referenced_doc_names(load_questions())
    present = [n for n in names if pdf_path(n).exists()]
    assert present, "no PDFs on disk; run `make fetch-data` first"
    smallest = min(present, key=lambda n: pdf_path(n).stat().st_size)

    result = ingest_docs([smallest], build_docling_parser(), parsed_dir=tmp_path)

    assert result.parsed == [smallest]
    assert result.failures == []
    document = json.loads(parsed_path(smallest, tmp_path).read_text(encoding="utf-8"))
    assert document["schema_name"] == "DoclingDocument"
    assert document["texts"], "expected extracted text items"
    first_prov = next(prov for item in document["texts"] for prov in item.get("prov", []))
    assert isinstance(first_prov["page_no"], int)
