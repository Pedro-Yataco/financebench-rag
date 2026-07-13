"""Tests for per-page text reconstruction and the evidence fuzzy matcher."""

from typing import Any

import pytest

from scripts.verify_evidence import match_score
from src.pages import evidence_page_no, page_texts


def _doc(
    *,
    texts: list[dict[str, Any]] | None = None,
    tables: list[dict[str, Any]] | None = None,
    groups: list[dict[str, Any]] | None = None,
    body: list[str] | None = None,
    furniture: list[str] | None = None,
    pages: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    """Minimal DoclingDocument-export-shaped dict."""
    return {
        "body": {"self_ref": "#/body", "children": [{"$ref": ref} for ref in body or []]},
        "furniture": {
            "self_ref": "#/furniture",
            "children": [{"$ref": ref} for ref in furniture or []],
        },
        "texts": texts or [],
        "tables": tables or [],
        "groups": groups or [],
        "pages": {str(no): {"page_no": no} for no in pages},
    }


def _text(text: str, provs: list[tuple[int, tuple[int, int]]]) -> dict[str, Any]:
    return {
        "children": [],
        "label": "text",
        "prov": [{"page_no": no, "charspan": list(span)} for no, span in provs],
        "text": text,
    }


def _cell(text: str, row: int, col: int) -> dict[str, Any]:
    return {"text": text, "start_row_offset_idx": row, "start_col_offset_idx": col}


def test_page_texts_joins_items_in_reading_order() -> None:
    doc = _doc(
        texts=[
            _text("first line", [(1, (0, 10))]),
            _text("second line", [(1, (0, 11))]),
        ],
        body=["#/texts/0", "#/texts/1"],
    )

    assert page_texts(doc) == {1: "first line\nsecond line"}


def test_page_texts_splits_multi_prov_text_by_charspan() -> None:
    doc = _doc(
        texts=[_text("ends here starts there", [(1, (0, 9)), (2, (10, 22))])],
        body=["#/texts/0"],
        pages=(1, 2),
    )

    assert page_texts(doc) == {1: "ends here", 2: "starts there"}


def test_page_texts_keeps_blank_pages_as_empty_strings() -> None:
    doc = _doc(
        texts=[_text("only page two", [(2, (0, 13))])],
        body=["#/texts/0"],
        pages=(1, 2, 3),
    )

    assert page_texts(doc) == {1: "", 2: "only page two", 3: ""}


def test_page_texts_reaches_texts_nested_in_groups() -> None:
    doc = _doc(
        texts=[_text("inside a group", [(1, (0, 14))])],
        groups=[
            {
                "self_ref": "#/groups/0",
                "children": [{"$ref": "#/texts/0"}],
                "label": "list",
            }
        ],
        body=["#/groups/0"],
    )

    assert page_texts(doc) == {1: "inside a group"}


def test_page_texts_includes_furniture_items() -> None:
    doc = _doc(
        texts=[
            _text("body text", [(1, (0, 9))]),
            _text("page header", [(1, (0, 11))]),
        ],
        body=["#/texts/0"],
        furniture=["#/texts/1"],
    )

    assert page_texts(doc) == {1: "body text\npage header"}


def test_page_texts_serializes_table_rows_once_per_spanning_cell() -> None:
    wide = _cell("Revenue", 0, 0)  # spans both columns: appears twice in the grid
    table = {
        "self_ref": "#/tables/0",
        "children": [],
        "label": "table",
        "prov": [{"page_no": 1, "charspan": [0, 0]}],
        "data": {
            "grid": [
                [wide, wide],
                [_cell("2018", 1, 0), _cell("$ 32,765", 1, 1)],
                [_cell("", 2, 0), _cell("", 2, 1)],
            ]
        },
    }
    doc = _doc(tables=[table], body=["#/tables/0"])

    assert page_texts(doc) == {1: "Revenue\n2018 $ 32,765"}


def test_evidence_page_no_maps_zero_based_annotation_to_docling_page() -> None:
    assert evidence_page_no(0) == 1
    assert evidence_page_no(41) == 42


def test_match_score_exact_text_is_one() -> None:
    text = "net sales increased 3.5 percent to 32.8 billion dollars in 2018"

    assert match_score(text, f"intro. {text} outro.") == 1.0


def test_match_score_ignores_case_punctuation_and_whitespace() -> None:
    needle = "Net sales increased 3.5% to $32.8 billion in 2018"
    haystack = "NET  SALES increased\n3.5 % to $ 32,8 billion in 2018 blah"

    # "32.8" vs "32,8" both tokenize to ["32", "8"], so this still matches fully
    assert match_score(needle, haystack) == 1.0


def test_match_score_absent_needle_is_zero() -> None:
    assert match_score("alpha beta gamma delta epsilon zeta", "totally unrelated words") == 0.0


def test_match_score_partial_overlap_is_fractional() -> None:
    # 6 tokens -> two 5-grams; the haystack contains only the first one
    needle = "one two three four five six"
    haystack = "one two three four five STOP six"

    assert match_score(needle, haystack) == 0.5


def test_match_score_short_needle_requires_token_boundaries() -> None:
    assert match_score("cat dog", "the cat dog runs") == 1.0
    assert match_score("cat dog", "the cat dogma runs") == 0.0


def test_match_score_empty_needle_raises() -> None:
    with pytest.raises(ValueError, match="no tokens"):
        match_score("...", "anything")
