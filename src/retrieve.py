"""Page retrieval over parsed documents.

Phase-1 baseline: classic BM25 (bm25s) over per-page plain text, one
in-memory index per document. Retrieval is always scoped to a single
doc_name — the oracle-doc setting (D-001) — so indexes are small and built
lazily from the parsed cache, then kept for the process lifetime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import bm25s

from src.ingest import PARSED_DIR
from src.pages import load_page_texts


class PageRetriever(Protocol):
    """Anything that ranks the pages of one document for a question."""

    def retrieve(self, question: str, doc_name: str, k: int) -> list[int]: ...


class BM25PageRetriever:
    """BM25 over page text with one lazily built index per doc."""

    def __init__(self, parsed_dir: Path = PARSED_DIR) -> None:
        self._parsed_dir = parsed_dir
        self._indexes: dict[str, tuple[bm25s.BM25, list[int]]] = {}

    def _index(self, doc_name: str) -> tuple[bm25s.BM25, list[int]]:
        if doc_name not in self._indexes:
            pages = load_page_texts(doc_name, self._parsed_dir)
            page_nos = sorted(pages)
            index = bm25s.BM25()
            corpus = [pages[no] for no in page_nos]
            tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
            index.index(tokens, show_progress=False)
            self._indexes[doc_name] = (index, page_nos)
        return self._indexes[doc_name]

    def retrieve(self, question: str, doc_name: str, k: int) -> list[int]:
        """Top-k page_nos of doc_name ranked by BM25 score for the question."""
        index, page_nos = self._index(doc_name)
        query = bm25s.tokenize(question, stopwords="en", show_progress=False)
        results, _ = index.retrieve(query, k=min(k, len(page_nos)), show_progress=False)
        return [page_nos[int(i)] for i in results[0]]
