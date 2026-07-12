"""Fetch the FinanceBench open subset into data/raw/ via git sparse checkout.

Clones patronus-ai/financebench without blobs, checks out the two question
JSONLs, then narrows the sparse pattern set to exactly the PDFs referenced by
the open-source questions. Re-running is cheap and restores missing files.
"""

from __future__ import annotations

import subprocess
import sys

from src.dataset import (
    RAW_DIR,
    load_doc_info,
    load_questions,
    pdf_path,
    referenced_doc_names,
    validate_doc_references,
)

REPO_URL = "https://github.com/patronus-ai/financebench.git"


def _git(*args: str, stdin: str | None = None) -> str:
    """Run git inside the clone, returning stdout; stderr streams through."""
    result = subprocess.run(
        ["git", "-C", str(RAW_DIR), *args],
        input=stdin,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout


def main() -> int:
    if not (RAW_DIR / ".git").exists():
        RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--depth",
                "1",
                REPO_URL,
                str(RAW_DIR),
            ],
            check=True,
        )
    branch = _git("symbolic-ref", "--short", "HEAD").strip()

    _git("sparse-checkout", "set", "--no-cone", "/data/*")
    _git("checkout", branch)

    questions = load_questions()
    doc_info = load_doc_info()
    validate_doc_references(questions, doc_info)
    doc_names = referenced_doc_names(questions)

    patterns = ["/data/*"] + [f"/pdfs/{name}.pdf" for name in doc_names]
    _git("sparse-checkout", "set", "--no-cone", "--stdin", stdin="\n".join(patterns) + "\n")
    _git("checkout", branch)

    missing = [name for name in doc_names if not pdf_path(name).exists()]
    present = len(doc_names) - len(missing)
    print(f"{len(questions)} questions / {len(doc_names)} unique docs / {present} PDFs present")
    if missing:
        print(f"missing PDFs: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
