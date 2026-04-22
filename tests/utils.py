"""Matching helpers + test-case types used by conftest.py and test_extraction.py.

All matchers return a (value, diagnostic) tuple. Tests assert on the value
and surface the message in the failure output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from models import Item, Scale, Survey


@dataclass
class ExpectedCase:
    """One YAML fixture under tests/expected/ — the unit of parametrization."""
    pdf: str
    path: Path
    spec: dict

    @property
    def id(self) -> str:
        return self.pdf


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace for substring matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_all(haystack: str, keywords: Iterable[str]) -> bool:
    h = normalize(haystack)
    return all(normalize(k) in h for k in keywords)


def contains_any(haystack: str, keywords: Iterable[str]) -> bool:
    h = normalize(haystack)
    return any(normalize(k) in h for k in keywords)


def walk_scales(scales: list[Scale]):
    """Depth-first traversal yielding every scale in the tree."""
    for s in scales:
        yield s
        yield from walk_scales(s.subscales)


def find_item_by_keywords(
    survey: Survey, keywords: list[str]
) -> tuple[Optional[Item], str]:
    """Return (item, diagnostic). Item is None unless exactly one item matches."""
    matches = [i for i in survey.items if contains_all(i.item_text, keywords)]
    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        candidates = "; ".join(f"#{i.item_id}: {i.item_text[:80]}" for i in survey.items[:10])
        return None, (
            f"no item matched keywords {keywords!r}. "
            f"First candidates: {candidates or '(no items in survey)'}"
        )
    matched = "; ".join(f"#{m.item_id}: {m.item_text[:80]}" for m in matches)
    return None, f"ambiguous match for {keywords!r} — {len(matches)} items matched: {matched}"


def scales_containing_item(survey: Survey, item_id: int) -> list[Scale]:
    """Every scale in the tree whose scored_items directly references this item_id."""
    return [
        s for s in walk_scales(survey.scales)
        if any(si.item_id == item_id for si in s.scored_items)
    ]


def scales_matching_keywords(survey: Survey, keywords_any_of: list[str]) -> list[Scale]:
    """Every scale in the tree whose scale_name contains any of the keywords."""
    return [s for s in walk_scales(survey.scales) if contains_any(s.scale_name, keywords_any_of)]


def check_range(value: int | float, spec: dict) -> tuple[bool, str]:
    """Evaluate a numeric spec of the form {min: N}, {max: N}, {min, max}, or {exact: N}."""
    if spec is None:
        return True, ""
    if "exact" in spec:
        ok = value == spec["exact"]
        return ok, f"got {value}, expected exactly {spec['exact']}"
    lo = spec.get("min")
    hi = spec.get("max")
    if lo is not None and value < lo:
        return False, f"got {value}, expected >= {lo}"
    if hi is not None and value > hi:
        return False, f"got {value}, expected <= {hi}"
    return True, ""


def detect_english_fraction(texts: list[str]) -> float:
    """Fraction of non-empty texts where langdetect classifies the text as English."""
    try:
        from langdetect import detect, DetectorFactory, LangDetectException
    except ImportError:  # pragma: no cover
        raise RuntimeError(
            "langdetect is required for language assertions. "
            "Install test dependencies: poetry install --with test"
        )

    DetectorFactory.seed = 0
    non_empty = [t for t in texts if t and t.strip()]
    if not non_empty:
        return 0.0
    english = 0
    for t in non_empty:
        try:
            if detect(t) == "en":
                english += 1
        except LangDetectException:
            pass
    return english / len(non_empty)
