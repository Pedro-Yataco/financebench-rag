"""Token-aware chunking of parsed documents with page and filing metadata.

Docling's HybridChunker splits each cached DoclingDocument along its layout
tree, capping chunks at MAX_TOKENS of the BGE-M3 tokenizer (the Phase-2
embedding model) and merging undersized peers under the same headings. Each
chunk is stored as one JSONL line carrying the text exactly as it will be
embedded — contextualize() output, headings included — plus the page numbers
its items came from, which downstream page-level metrics consume (D-016).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from docling_core.transforms.chunker import BaseChunker, DocChunk, HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.base import BaseDocSerializer
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import TableItem
from pydantic import BaseModel

from src.dataset import DocInfo, load_doc_info, load_questions, referenced_doc_names
from src.ingest import PARSED_DIR, parsed_path

CHUNKS_DIR = Path("data/chunks")
TOKENIZER_MODEL_ID = "BAAI/bge-m3"
MAX_TOKENS = 512  # initial value; the P2.T6 ablation revisits it (D-016)


class Chunk(BaseModel):
    """One retrievable unit: embedded text plus its provenance and metadata."""

    chunk_id: str
    doc_name: str
    company: str
    year: int
    doc_type: str
    pages: list[int]
    text: str


def chunks_path(doc_name: str, chunks_dir: Path = CHUNKS_DIR) -> Path:
    """JSONL file holding the chunks of one doc."""
    return chunks_dir / f"{doc_name}.jsonl"


def chunk_pages(chunk: DocChunk) -> list[int]:
    """Sorted unique page_nos over every prov of every doc item in the chunk."""
    return sorted({prov.page_no for item in chunk.meta.doc_items for prov in item.prov})


class _CappedTokenizer(BaseTokenizer):
    """Delegates token counting but reports a smaller budget.

    LineBasedTokenChunker takes its budget from tokenizer.get_max_tokens();
    this is the only channel through which a partial budget can reach it.
    """

    inner: BaseTokenizer
    cap: int

    def count_tokens(self, text: str) -> int:
        return self.inner.count_tokens(text)

    def get_max_tokens(self) -> int:
        return self.cap

    def get_tokenizer(self) -> Any:
        return self.inner.get_tokenizer()


class BudgetedHybridChunker(HybridChunker):
    """HybridChunker that actually enforces the budget on row-split tables.

    docling-core 2.87.0 passes max_tokens=available_length when building the
    LineBasedTokenChunker for oversized tables, but that chunker has no such
    field: pydantic silently drops the kwarg and the full tokenizer budget is
    used for the table alone, so the final chunk (headings included) overruns
    max_tokens by the heading length. Only the table branch of segment() is
    replaced; everything else defers to the parent.
    """

    def segment(
        self, doc_chunk: DocChunk, available_length: int, doc_serializer: BaseDocSerializer
    ) -> list[str]:
        if (
            self.repeat_table_header
            and isinstance(doc_serializer, ChunkingDocSerializer)
            and len(doc_chunk.meta.doc_items) == 1
            and isinstance(doc_chunk.meta.doc_items[0], TableItem)
        ):
            header_lines, body_lines = doc_serializer.table_serializer.get_header_and_body_lines(
                table_text=doc_chunk.text
            )
            line_chunker = LineBasedTokenChunker(
                tokenizer=_CappedTokenizer(inner=self.tokenizer, cap=available_length),
                prefix="\n".join(header_lines),
                omit_prefix_on_overflow=self.omit_header_on_overflow,
                serializer_provider=self.serializer_provider,
            )
            return line_chunker.chunk_text(lines=body_lines)
        return super().segment(doc_chunk, available_length, doc_serializer)


def build_hybrid_chunker(max_tokens: int = MAX_TOKENS) -> BudgetedHybridChunker:
    """Production chunker: token budget aligned with the embedding model.

    Downloads the BGE-M3 tokenizer from Hugging Face on first use.
    """
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name=TOKENIZER_MODEL_ID, max_tokens=max_tokens
    )
    return BudgetedHybridChunker(tokenizer=tokenizer)


def build_chunks(doc: DoclingDocument, info: DocInfo, chunker: BaseChunker) -> list[Chunk]:
    """Chunk one document, attaching filing metadata and page provenance."""
    chunks = []
    for i, base_chunk in enumerate(chunker.chunk(dl_doc=doc)):
        if not isinstance(base_chunk, DocChunk):
            raise TypeError(f"expected DocChunk, got {type(base_chunk).__name__}")
        chunks.append(
            Chunk(
                chunk_id=f"{info.doc_name}:{i:04d}",
                doc_name=info.doc_name,
                company=info.company,
                year=info.doc_period,
                doc_type=info.doc_type,
                pages=chunk_pages(base_chunk),
                text=chunker.contextualize(base_chunk),
            )
        )
    return chunks


def write_chunks(chunks: Sequence[Chunk], path: Path) -> None:
    """One JSON line per chunk, written atomically (tmp file + replace)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(f"{chunk.model_dump_json()}\n" for chunk in chunks), encoding="utf-8")
    tmp.replace(path)


def load_chunks(path: Path) -> list[Chunk]:
    """Parse every chunk record in a JSONL file."""
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _log_flushed(message: str) -> None:
    print(message, flush=True)


def chunk_docs(
    doc_names: Sequence[str],
    doc_info: Mapping[str, DocInfo],
    chunker: BaseChunker,
    parsed_dir: Path = PARSED_DIR,
    chunks_dir: Path = CHUNKS_DIR,
    limit: int | None = None,
    log: Callable[[str], None] = _log_flushed,
) -> tuple[list[str], list[str]]:
    """Chunk every uncached doc in order; returns (built, skipped) doc_names.

    A missing parsed cache raises: chunking presumes ingest completed.
    """
    todo = list(doc_names)[:limit] if limit is not None else list(doc_names)
    built: list[str] = []
    skipped: list[str] = []
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(todo, start=1):
        target = chunks_path(name, chunks_dir)
        prefix = f"[{i}/{len(todo)}] {name}"
        if target.exists():
            skipped.append(name)
            log(f"{prefix}: skipped (cached)")
            continue
        start = time.perf_counter()
        doc = DoclingDocument.load_from_json(parsed_path(name, parsed_dir))
        chunks = build_chunks(doc, doc_info[name], chunker)
        write_chunks(chunks, target)
        log(f"{prefix}: {len(chunks)} chunks in {time.perf_counter() - start:.1f}s")
        built.append(name)
    return built, skipped


def histogram_lines(counts: Sequence[int], bin_size: int) -> list[str]:
    """One line per token bin from 0 to the max count, empty bins included."""
    if not counts:
        return []
    bins = [0] * (max(counts) // bin_size + 1)
    for count in counts:
        bins[count // bin_size] += 1
    return [f"{i * bin_size:4d}-{(i + 1) * bin_size - 1:4d}: {n}" for i, n in enumerate(bins)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk parsed docs into data/chunks/.")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="process only the first N docs"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS, help="chunk token budget (BGE-M3 tokens)"
    )
    args = parser.parse_args(argv)

    doc_names = referenced_doc_names(load_questions())
    doc_info = load_doc_info()
    chunker = build_hybrid_chunker(args.max_tokens)
    print(f"chunker: {TOKENIZER_MODEL_ID} tokenizer, max_tokens={args.max_tokens}", flush=True)

    built, skipped = chunk_docs(doc_names, doc_info, chunker, limit=args.limit)
    print(f"{len(built)} chunked / {len(skipped)} skipped (cached)")

    files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    all_chunks = [chunk for file in files for chunk in load_chunks(file)]
    token_counts = [chunker.tokenizer.count_tokens(chunk.text) for chunk in all_chunks]
    print(
        f"{len(all_chunks)} chunks over {len(files)} docs | tokens min {min(token_counts)}"
        f" / mean {sum(token_counts) / len(token_counts):.0f} / max {max(token_counts)}"
    )
    for line in histogram_lines(token_counts, bin_size=64):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
