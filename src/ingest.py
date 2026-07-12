"""Docling ingest CLI: parse every referenced PDF into a per-doc JSON cache.

Each doc referenced by the open-source questions is parsed once and cached as
data/parsed/<doc_name>.json (a DoclingDocument export). Cached docs are
skipped on re-runs, so the multi-hour full-corpus run is resumable; per-doc
failures are collected and reported instead of aborting the run.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dataset import load_questions, pdf_path, referenced_doc_names

PARSED_DIR = Path("data/parsed")

ParseFn = Callable[[Path], dict[str, Any]]


@dataclass
class IngestResult:
    """Outcome of one ingest run, in processing order."""

    parsed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)


def parsed_path(doc_name: str, parsed_dir: Path = PARSED_DIR) -> Path:
    """Cache file for one parsed doc."""
    return parsed_dir / f"{doc_name}.json"


def _log_flushed(message: str) -> None:
    print(message, flush=True)


def _write_atomic(target: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temp file so an interrupted run never leaves a partial cache."""
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def ingest_docs(
    doc_names: Sequence[str],
    parse: ParseFn,
    parsed_dir: Path = PARSED_DIR,
    limit: int | None = None,
    log: Callable[[str], None] = _log_flushed,
) -> IngestResult:
    """Parse every uncached doc in order, caching one JSON per doc."""
    todo = list(doc_names)[:limit] if limit is not None else list(doc_names)
    result = IngestResult()
    parsed_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(todo, start=1):
        target = parsed_path(name, parsed_dir)
        prefix = f"[{i}/{len(todo)}] {name}"
        if target.exists():
            result.skipped.append(name)
            log(f"{prefix}: skipped (cached)")
            continue
        start = time.perf_counter()
        try:
            document = parse(pdf_path(name))
        except Exception as exc:  # one bad PDF must not abort the overnight run
            result.failures.append((name, f"{type(exc).__name__}: {exc}"))
            log(f"{prefix}: FAILED ({type(exc).__name__}: {exc})")
            continue
        _write_atomic(target, document)
        log(f"{prefix}: parsed in {time.perf_counter() - start:.1f}s")
        result.parsed.append(name)
    return result


def format_summary(result: IngestResult) -> str:
    """One summary line plus one line per failure."""
    lines = [
        f"{len(result.parsed)} parsed / {len(result.skipped)} skipped (cached) / "
        f"{len(result.failures)} failed"
    ]
    lines.extend(f"  FAILED {name}: {error}" for name, error in result.failures)
    return "\n".join(lines)


def build_docling_parser() -> ParseFn:
    """Build the real Docling parse function (imports deferred: heavy deps)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    # SEC filings are born-digital; OCR would only slow the run down. Docs that
    # turn out to need it are re-parsed individually during failure triage.
    options.do_ocr = False
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    def parse(pdf: Path) -> dict[str, Any]:
        document: dict[str, Any] = converter.convert(pdf).document.export_to_dict()
        return document

    return parse


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse referenced PDFs into data/parsed/.")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="process only the first N docs"
    )
    args = parser.parse_args(argv)

    doc_names = referenced_doc_names(load_questions())
    start = time.perf_counter()
    result = ingest_docs(doc_names, build_docling_parser(), limit=args.limit)
    elapsed = time.perf_counter() - start
    print(format_summary(result))
    print(f"total {elapsed / 60:.1f} min", flush=True)
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
