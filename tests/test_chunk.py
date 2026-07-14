"""Tests for src.chunk: page mapping, metadata propagation, store round-trip.

The unit tests run the real HybridChunker against a synthetic DoclingDocument
with a whitespace tokenizer, so docling's chunking plumbing is exercised
without downloading models. The BGE-M3 tokenizer and the parsed cache only
appear in the integration test.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    BaseMeta,
    DocChunk,
    DocMeta,
    HybridChunker,
)
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    TableCell,
    TableData,
)
from docling_core.types.doc.document import DocItem, TextItem

from src.chunk import (
    BudgetedHybridChunker,
    Chunk,
    build_chunks,
    chunk_docs,
    chunk_pages,
    chunks_path,
    histogram_lines,
    load_chunks,
    write_chunks,
)
from src.dataset import DocInfo

# Small budget so the synthetic doc exercises both the split path (oversized
# paragraph) and the no-merge path (neighbors that would fuse pages 1 and 2).
SMALL_MAX_TOKENS = 16

DOC_NAME = "TEST_2020_10K"
DOC_INFO = DocInfo(
    doc_name=DOC_NAME,
    company="TestCo",
    gics_sector="Industrials",
    doc_type="10k",
    doc_period=2020,
    doc_link="https://example.com/TEST_2020_10K.pdf",
)


class WordTokenizer(BaseTokenizer):
    """Whitespace token counter, so unit tests never download a tokenizer."""

    max_tokens: int = SMALL_MAX_TOKENS

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return self.count_tokens


class FakeChunker(BaseChunker):
    """Yields pre-built chunks, letting tests control DocChunk internals."""

    chunks: list[DocChunk]

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        return iter(self.chunks)


def prov(page_no: int, text: str = "") -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no, bbox=BoundingBox(l=0, t=0, r=100, b=100), charspan=(0, len(text))
    )


def text_item(ref_idx: int, text: str, pages: list[int]) -> TextItem:
    return TextItem(
        self_ref=f"#/texts/{ref_idx}",
        label=DocItemLabel.TEXT,
        orig=text,
        text=text,
        prov=[prov(page, text) for page in pages],
    )


def doc_chunk(
    text: str, pages_per_item: list[list[int]], headings: list[str] | None = None
) -> DocChunk:
    items: list[DocItem] = [text_item(i, text, pages) for i, pages in enumerate(pages_per_item)]
    return DocChunk(text=text, meta=DocMeta(doc_items=items, headings=headings))


def synthetic_doc() -> DoclingDocument:
    """Two-page document: heading + short/long/page-spanning text + a table."""
    doc = DoclingDocument(name=DOC_NAME)
    heading = "Results of Operations"
    doc.add_text(label=DocItemLabel.SECTION_HEADER, text=heading, prov=prov(1, heading))
    short = "Net sales grew nine percent to a record level this year."
    doc.add_text(label=DocItemLabel.TEXT, text=short, prov=prov(1, short))
    long = " ".join(f"Segment {i} recorded strong operating income growth." for i in range(20))
    doc.add_text(label=DocItemLabel.TEXT, text=long, prov=prov(1, long))
    spanning = "Cash flow discussion continues across the page break here."
    spanning_item = doc.add_text(label=DocItemLabel.TEXT, text=spanning, prov=prov(1, spanning))
    spanning_item.prov.append(prov(2, spanning))
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
            column_header=row == 0,
            row_header=row > 0 and col == 0,
        )
        for row, col, text in [
            (0, 0, "Metric"),
            (0, 1, "FY2020"),
            (0, 2, "FY2019"),
            (1, 0, "Revenue"),
            (1, 1, "32136"),
            (1, 2, "32765"),
        ]
    ]
    doc.add_table(data=TableData(table_cells=cells, num_rows=2, num_cols=3), prov=prov(2))
    return doc


class TestChunkPages:
    def test_sorted_unique_across_items_and_provs(self) -> None:
        chunk = doc_chunk("x", pages_per_item=[[3], [2, 3], [2]])
        assert chunk_pages(chunk) == [2, 3]

    def test_item_without_prov_contributes_nothing(self) -> None:
        item = TextItem(self_ref="#/texts/0", label=DocItemLabel.TEXT, orig="x", text="x")
        chunk = DocChunk(text="x", meta=DocMeta(doc_items=[item]))
        assert chunk_pages(chunk) == []


class TestBuildChunks:
    def test_metadata_and_deterministic_ids(self) -> None:
        prebuilt = [
            doc_chunk("Cash was $5 million.", [[2]]),
            doc_chunk("Debt was $9 million.", [[5], [4]]),
        ]
        built = build_chunks(DoclingDocument(name=DOC_NAME), DOC_INFO, FakeChunker(chunks=prebuilt))
        assert [chunk.chunk_id for chunk in built] == [f"{DOC_NAME}:0000", f"{DOC_NAME}:0001"]
        first = built[0]
        assert first.doc_name == DOC_NAME
        assert first.company == "TestCo"
        assert first.year == 2020
        assert first.doc_type == "10k"
        assert first.pages == [2]
        assert built[1].pages == [4, 5]

    def test_text_is_contextualized_with_headings(self) -> None:
        prebuilt = [doc_chunk("Cash was $5 million.", [[2]], headings=["Note 5. Cash"])]
        built = build_chunks(DoclingDocument(name=DOC_NAME), DOC_INFO, FakeChunker(chunks=prebuilt))
        assert built[0].text == "Note 5. Cash\nCash was $5 million."

    def test_rejects_chunks_without_doc_meta(self) -> None:
        class BareChunker(BaseChunker):
            def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
                return iter([BaseChunk(text="x", meta=BaseMeta())])

        with pytest.raises(TypeError, match="DocChunk"):
            build_chunks(DoclingDocument(name=DOC_NAME), DOC_INFO, BareChunker())


def overflow_table_doc() -> DoclingDocument:
    """A table too large for the budget, under a heading: the row-split path."""
    doc = DoclingDocument(name=DOC_NAME)
    heading = "Notes to the Financial Statements"
    doc.add_text(label=DocItemLabel.SECTION_HEADER, text=heading, prov=prov(1, heading))
    header = ["Metric", "FY2020", "FY2019", "FY2018"]
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
            column_header=True,
        )
        for col, text in enumerate(header)
    ]
    for row in range(1, 5):
        values = [f"Line item {row}", f"1,{row}11", f"2,{row}22", f"3,{row}33"]
        cells.extend(
            TableCell(
                text=text,
                start_row_offset_idx=row,
                end_row_offset_idx=row + 1,
                start_col_offset_idx=col,
                end_col_offset_idx=col + 1,
                row_header=col == 0,
            )
            for col, text in enumerate(values)
        )
    doc.add_table(data=TableData(table_cells=cells, num_rows=5, num_cols=4), prov=prov(1))
    return doc


class TestHybridChunking:
    @pytest.fixture
    def built(self) -> list[Chunk]:
        chunker = BudgetedHybridChunker(tokenizer=WordTokenizer())
        return build_chunks(synthetic_doc(), DOC_INFO, chunker)

    def test_every_chunk_is_within_the_token_budget(self, built: list[Chunk]) -> None:
        tokenizer = WordTokenizer()
        assert built
        assert all(tokenizer.count_tokens(chunk.text) <= SMALL_MAX_TOKENS for chunk in built)

    def test_pages_stay_within_the_document(self, built: list[Chunk]) -> None:
        assert all(chunk.pages and set(chunk.pages) <= {1, 2} for chunk in built)

    def test_oversized_text_splits_into_same_page_chunks(self, built: list[Chunk]) -> None:
        pieces = [chunk for chunk in built if "Segment" in chunk.text]
        assert len(pieces) >= 2
        assert all(piece.pages == [1] for piece in pieces)

    def test_page_spanning_item_reports_both_pages(self, built: list[Chunk]) -> None:
        spanning = [chunk for chunk in built if "page break" in chunk.text]
        assert spanning and spanning[0].pages == [1, 2]

    def test_table_stays_intact_in_one_chunk(self, built: list[Chunk]) -> None:
        with_value = [chunk for chunk in built if "32136" in chunk.text]
        assert len(with_value) == 1
        table_chunk = with_value[0]
        assert "32765" in table_chunk.text
        assert table_chunk.pages == [2]

    def test_chunking_is_deterministic(self, built: list[Chunk]) -> None:
        again = build_chunks(
            synthetic_doc(), DOC_INFO, BudgetedHybridChunker(tokenizer=WordTokenizer())
        )
        assert built == again

    def test_row_split_table_chunks_respect_budget_including_headings(self) -> None:
        chunker = BudgetedHybridChunker(tokenizer=WordTokenizer())
        built = build_chunks(overflow_table_doc(), DOC_INFO, chunker)
        tokenizer = WordTokenizer()
        assert len([chunk for chunk in built if "FY2020" in chunk.text]) >= 2
        assert all(tokenizer.count_tokens(chunk.text) <= SMALL_MAX_TOKENS for chunk in built)

    def test_upstream_chunker_still_overruns_budget_on_split_tables(self) -> None:
        """Canary: once this fails, docling fixed the silently dropped
        max_tokens kwarg in segment() and BudgetedHybridChunker can be retired."""
        built = build_chunks(
            overflow_table_doc(), DOC_INFO, HybridChunker(tokenizer=WordTokenizer())
        )
        tokenizer = WordTokenizer()
        assert any(tokenizer.count_tokens(chunk.text) > SMALL_MAX_TOKENS for chunk in built)


class TestChunkStore:
    def test_write_and_load_roundtrip(self, tmp_path: Path) -> None:
        chunks = [
            Chunk(
                chunk_id=f"{DOC_NAME}:0000",
                doc_name=DOC_NAME,
                company="TestCo",
                year=2020,
                doc_type="10k",
                pages=[1, 2],
                text="Line one.\nLine two.",
            )
        ]
        path = tmp_path / f"{DOC_NAME}.jsonl"
        write_chunks(chunks, path)
        assert load_chunks(path) == chunks


class TestChunkDocs:
    @pytest.fixture
    def parsed_dir(self, tmp_path: Path) -> Path:
        directory = tmp_path / "parsed"
        directory.mkdir()
        export = synthetic_doc().export_to_dict()
        (directory / f"{DOC_NAME}.json").write_text(json.dumps(export), encoding="utf-8")
        return directory

    def test_builds_then_skips_cached(self, parsed_dir: Path, tmp_path: Path) -> None:
        chunks_dir = tmp_path / "chunks"
        chunker = BudgetedHybridChunker(tokenizer=WordTokenizer())
        infos = {DOC_NAME: DOC_INFO}
        logs: list[str] = []

        built, skipped = chunk_docs(
            [DOC_NAME],
            infos,
            chunker,
            parsed_dir=parsed_dir,
            chunks_dir=chunks_dir,
            log=logs.append,
        )
        assert (built, skipped) == ([DOC_NAME], [])
        stored = load_chunks(chunks_path(DOC_NAME, chunks_dir))
        assert stored
        assert {chunk.doc_name for chunk in stored} == {DOC_NAME}
        assert any("32136" in chunk.text for chunk in stored)

        built, skipped = chunk_docs(
            [DOC_NAME],
            infos,
            chunker,
            parsed_dir=parsed_dir,
            chunks_dir=chunks_dir,
            log=logs.append,
        )
        assert (built, skipped) == ([], [DOC_NAME])
        assert any("skipped (cached)" in line for line in logs)

    def test_limit_processes_only_the_first_docs(self, parsed_dir: Path, tmp_path: Path) -> None:
        chunks_dir = tmp_path / "chunks"
        chunker = BudgetedHybridChunker(tokenizer=WordTokenizer())
        built, skipped = chunk_docs(
            [DOC_NAME, "MISSING_DOC"],
            {DOC_NAME: DOC_INFO},
            chunker,
            parsed_dir=parsed_dir,
            chunks_dir=chunks_dir,
            limit=1,
            log=lambda _: None,
        )
        assert (built, skipped) == ([DOC_NAME], [])

    def test_missing_parsed_cache_raises(self, parsed_dir: Path, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            chunk_docs(
                ["MISSING_DOC"],
                {},
                BudgetedHybridChunker(tokenizer=WordTokenizer()),
                parsed_dir=parsed_dir,
                chunks_dir=tmp_path / "chunks",
                log=lambda _: None,
            )


class TestHistogramLines:
    def test_bins_cover_the_full_range_including_empty_ones(self) -> None:
        assert histogram_lines([10, 70, 70, 200], bin_size=64) == [
            "   0-  63: 1",
            "  64- 127: 2",
            " 128- 191: 0",
            " 192- 255: 1",
        ]

    def test_empty_counts_produce_no_lines(self) -> None:
        assert histogram_lines([], bin_size=64) == []


@pytest.mark.integration
class TestRealTokenizerChunking:
    # Docling's line splitter budgets long table rows by summing per-segment
    # token counts, which drifts a few tokens from the count of the joined
    # text under SentencePiece; harmless for embedding (BGE-M3 takes 8192).
    TOKEN_SLACK = 16

    def test_smallest_parsed_doc_chunks_within_budget(self) -> None:
        from src.chunk import MAX_TOKENS, build_hybrid_chunker
        from src.dataset import load_doc_info
        from src.ingest import PARSED_DIR

        parsed_files = sorted(PARSED_DIR.glob("*.json"), key=lambda p: p.stat().st_size)
        if not parsed_files:
            pytest.skip("no parsed cache under data/parsed")
        smallest = parsed_files[0]
        info = load_doc_info()[smallest.stem]

        chunker = build_hybrid_chunker()
        doc = DoclingDocument.load_from_json(smallest)
        chunks = build_chunks(doc, info, chunker)

        assert chunks
        page_nos = {int(no) for no in doc.pages}
        tokenizer = chunker.tokenizer
        for chunk in chunks:
            assert tokenizer.count_tokens(chunk.text) <= MAX_TOKENS + self.TOKEN_SLACK
            assert set(chunk.pages) <= page_nos
