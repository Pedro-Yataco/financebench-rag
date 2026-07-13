"""Verify FinanceBench evidence-page annotations against parsed page text.

For every evidence item, fuzzy-match its evidence_text against the text of
the annotated page under both indexing hypotheses — evidence_page_num being
0-based (Docling page_no = num + 1) or 1-based (page_no = num) — then decide
the indexing empirically and list every mismatch under the winning one, with
the best-matching page in the document as a diagnostic.

Exits non-zero if fewer than MIN_MATCH_RATE of the evidence items match on
their annotated page under the chosen indexing.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow running by file path

from src.dataset import load_questions  # noqa: E402
from src.pages import load_page_texts  # noqa: E402

NGRAM = 5
MATCH_THRESHOLD = 0.6
MIN_MATCH_RATE = 0.9

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def match_score(needle: str, haystack: str, n: int = NGRAM) -> float:
    """Fraction of the needle's word n-grams that appear in the haystack.

    Falls back to consecutive-token containment when the needle is shorter
    than n tokens. Tokens are lowercased alphanumeric runs, so punctuation
    and whitespace differences between the two texts do not matter.
    """
    needle_tokens = _tokens(needle)
    if not needle_tokens:
        raise ValueError("needle has no tokens")
    haystack_tokens = _tokens(haystack)
    if len(needle_tokens) < n:
        width = len(needle_tokens)
        found = any(
            haystack_tokens[i : i + width] == needle_tokens
            for i in range(len(haystack_tokens) - width + 1)
        )
        return float(found)
    needle_grams = _ngrams(needle_tokens, n)
    return len(needle_grams & _ngrams(haystack_tokens, n)) / len(needle_grams)


@dataclass
class EvidenceCheck:
    """Match scores for one evidence item under both indexing hypotheses."""

    financebench_id: str
    doc_name: str
    evidence_page_num: int
    score_same: float  # page_no = evidence_page_num      (1-based hypothesis)
    score_plus_one: float  # page_no = evidence_page_num + 1  (0-based hypothesis)
    best_page_no: int | None = None  # best-matching page, for mismatch diagnostics
    best_score: float = 0.0

    def score(self, offset: int) -> float:
        return self.score_plus_one if offset == 1 else self.score_same


def _check_doc(doc_name: str, items: list[tuple[str, int, str]]) -> list[EvidenceCheck]:
    """Score every (question_id, evidence_page_num, evidence_text) of one doc."""
    pages = load_page_texts(doc_name)
    checks = []
    for question_id, page_num, evidence_text in items:
        score_same = match_score(evidence_text, pages.get(page_num, ""))
        score_plus_one = match_score(evidence_text, pages.get(page_num + 1, ""))
        check = EvidenceCheck(question_id, doc_name, page_num, score_same, score_plus_one)
        if min(score_same, score_plus_one) < MATCH_THRESHOLD:
            scored = [(match_score(evidence_text, text), no) for no, text in pages.items() if text]
            check.best_score, check.best_page_no = max(scored, default=(0.0, None))
        checks.append(check)
    return checks


def main() -> int:
    questions = load_questions()
    by_doc: dict[str, list[tuple[str, int, str]]] = {}
    for question in questions:
        for item in question.evidence:
            by_doc.setdefault(question.doc_name, []).append(
                (question.financebench_id, item.evidence_page_num, item.evidence_text)
            )

    checks: dict[str, list[EvidenceCheck]] = {}
    for doc_name in sorted(by_doc):
        for check in _check_doc(doc_name, by_doc[doc_name]):
            checks.setdefault(check.financebench_id, []).append(check)

    ordered = [check for question in questions for check in checks[question.financebench_id]]
    total = len(ordered)
    matched_plus_one = sum(check.score_plus_one >= MATCH_THRESHOLD for check in ordered)
    matched_same = sum(check.score_same >= MATCH_THRESHOLD for check in ordered)
    offset = 1 if matched_plus_one >= matched_same else 0
    matched = max(matched_plus_one, matched_same)

    for check in ordered:
        verdict = "ok" if check.score(offset) >= MATCH_THRESHOLD else "MISS"
        print(
            f"{verdict:4} {check.financebench_id}  {check.doc_name}"
            f"  p{check.evidence_page_num}->page_no {check.evidence_page_num + offset}"
            f"  score={check.score(offset):.2f}"
        )

    print(f"\n{total} evidence items over {len(by_doc)} docs")
    print(f"0-based hypothesis (page_no = num + 1): {matched_plus_one}/{total} matched")
    print(f"1-based hypothesis (page_no = num):     {matched_same}/{total} matched")
    print(f"chosen indexing: {'0-based, offset +1' if offset == 1 else '1-based, offset 0'}")
    print(f"match rate: {matched / total:.1%} (threshold {MATCH_THRESHOLD}, n-gram {NGRAM})")

    misses = [check for check in ordered if check.score(offset) < MATCH_THRESHOLD]
    if misses:
        print(f"\nmismatches under chosen indexing ({len(misses)}):")
        for check in misses:
            best = (
                f"best page_no {check.best_page_no} score={check.best_score:.2f}"
                if check.best_page_no is not None
                else "no page matches at all"
            )
            print(
                f"  {check.financebench_id}  {check.doc_name}"
                f"  annotated page_no {check.evidence_page_num + offset}"
                f"  score={check.score(offset):.2f}  ({best})"
            )

    return 0 if matched / total >= MIN_MATCH_RATE else 1


if __name__ == "__main__":
    sys.exit(main())
