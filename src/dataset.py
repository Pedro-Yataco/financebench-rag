"""FinanceBench open-subset loading and validation.

The upstream repo (patronus-ai/financebench) ships two JSONL files: the 150
open-source questions and per-document metadata. This module defines their
schemas and typed loaders; scripts/fetch_data.py materializes the files under
data/raw/.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

RAW_DIR = Path("data/raw")
QUESTIONS_PATH = RAW_DIR / "data" / "financebench_open_source.jsonl"
DOC_INFO_PATH = RAW_DIR / "data" / "financebench_document_information.jsonl"
PDF_DIR = RAW_DIR / "pdfs"

QuestionType = Literal["metrics-generated", "domain-relevant", "novel-generated"]


class EvidenceItem(BaseModel):
    """One gold evidence annotation: a text span and the page it appears on."""

    doc_name: str
    evidence_page_num: int
    evidence_text: str
    evidence_text_full_page: str


class Question(BaseModel):
    """One open-subset question with its gold answer and evidence pages."""

    financebench_id: str
    company: str
    doc_name: str
    question_type: QuestionType
    question_reasoning: str | None
    domain_question_num: str | None
    question: str
    answer: str
    justification: str | None
    dataset_subset_label: Literal["OPEN_SOURCE"]
    evidence: list[EvidenceItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _evidence_doc_matches_question_doc(self) -> Question:
        for item in self.evidence:
            if item.doc_name != self.doc_name:
                raise ValueError(
                    f"evidence doc_name {item.doc_name!r} does not match "
                    f"question doc_name {self.doc_name!r}"
                )
        return self


class DocInfo(BaseModel):
    """Metadata for one source document (SEC filing or earnings report)."""

    doc_name: str
    company: str
    gics_sector: str
    doc_type: str
    doc_period: int
    doc_link: str


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Parse and validate every question record in a JSONL file."""
    return [
        Question.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_doc_info(path: Path = DOC_INFO_PATH) -> dict[str, DocInfo]:
    """Parse document metadata into a dict keyed by doc_name.

    The upstream file duplicates one doc_name (FOOTLOCKER_2023_annualreport)
    with conflicting doc_period values; the first occurrence wins.
    """
    infos: dict[str, DocInfo] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        info = DocInfo.model_validate_json(line)
        infos.setdefault(info.doc_name, info)
    return infos


def referenced_doc_names(questions: Iterable[Question]) -> list[str]:
    """Sorted unique doc_names referenced by the given questions."""
    return sorted({q.doc_name for q in questions})


def validate_doc_references(questions: Iterable[Question], doc_info: dict[str, DocInfo]) -> None:
    """Raise ValueError if any question references a doc_name absent from doc_info."""
    missing = sorted({q.doc_name for q in questions} - doc_info.keys())
    if missing:
        raise ValueError(f"questions reference doc_names missing from doc info: {missing}")


def pdf_path(doc_name: str, pdf_dir: Path = PDF_DIR) -> Path:
    """Filesystem path where the PDF for doc_name lives after fetch."""
    return pdf_dir / f"{doc_name}.pdf"
