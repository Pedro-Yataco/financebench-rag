"""Per-page plain text reconstructed from parsed Docling documents.

The ingest cache stores one DoclingDocument export per doc (see src.ingest).
This module rebuilds a plain-text view of every page: items are visited in
reading order (body tree, then furniture), text items contribute the charspan
slice of each prov to that prov's page, and tables are serialized row by row.
Every page of the document is present in the result, blank pages as "".
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.ingest import PARSED_DIR, parsed_path

# FinanceBench `evidence_page_num` is 0-based while Docling `page_no` is
# 1-based; verified empirically by scripts/verify_evidence.py (D-015).
GOLD_PAGE_OFFSET = 1


def evidence_page_no(evidence_page_num: int) -> int:
    """Docling page_no for a FinanceBench evidence_page_num annotation."""
    return evidence_page_num + GOLD_PAGE_OFFSET


def _resolve(doc: dict[str, Any], ref: str) -> dict[str, Any]:
    """Resolve a '#/texts/12'-style self-ref into its item dict."""
    _, kind, index = ref.split("/")
    item: dict[str, Any] = doc[kind][int(index)]
    return item


def _walk(doc: dict[str, Any], item: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield item
    for child in item.get("children", []):
        yield from _walk(doc, _resolve(doc, child["$ref"]))


def iter_items(doc: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """All content items in reading order: body tree first, then furniture."""
    for root in ("body", "furniture"):
        for child in doc[root].get("children", []):
            yield from _walk(doc, _resolve(doc, child["$ref"]))


def table_text(item: dict[str, Any]) -> str:
    """Serialize a table grid row by row; spanning cells contribute once."""
    rows = []
    for row_idx, row in enumerate(item["data"]["grid"]):
        cells = [
            cell["text"]
            for col_idx, cell in enumerate(row)
            if cell["start_row_offset_idx"] == row_idx
            and cell["start_col_offset_idx"] == col_idx
            and cell["text"]
        ]
        if cells:
            rows.append(" ".join(cells))
    return "\n".join(rows)


def page_texts(doc: dict[str, Any]) -> dict[int, str]:
    """Plain text keyed by page_no, covering every page of the document."""
    segments: dict[int, list[str]] = {int(no): [] for no in doc["pages"]}
    for item in iter_items(doc):
        if "text" in item:
            for prov in item["prov"]:
                start, end = prov["charspan"]
                segment = item["text"][start:end].strip()
                if segment:
                    segments.setdefault(prov["page_no"], []).append(segment)
        elif isinstance(item.get("data"), dict) and "grid" in item["data"]:
            text = table_text(item)
            if text:
                for prov in item["prov"]:
                    segments.setdefault(prov["page_no"], []).append(text)
    return {no: "\n".join(parts) for no, parts in segments.items()}


def load_page_texts(doc_name: str, parsed_dir: Path = PARSED_DIR) -> dict[int, str]:
    """Per-page text for one doc from the parsed cache."""
    doc = json.loads(parsed_path(doc_name, parsed_dir).read_text(encoding="utf-8"))
    return page_texts(doc)
