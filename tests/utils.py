"""Helpers + case type used by conftest.py and test_extraction.py.

The structural assertion is an unordered tree-isomorphism check: both the
expected spec and the extracted `Survey.scales` are reduced to the same
canonical nested-tuple form, so sibling order and scale names never leak
into the comparison.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models import Scale, Survey


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
    """Lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def walk_scales(scales: list[Scale]):
    """Depth-first traversal yielding every scale in the tree."""
    for s in scales:
        yield s
        yield from walk_scales(s.subscales)


# ---------------------------------------------------------------------------
# Structural canonicalization
# ---------------------------------------------------------------------------
#
# A canonical node is (own_items_count, tuple(sorted(canonical(child) for child in subscales))).
# Sorting the children at each level makes the form order-insensitive.
# Two trees are structurally equivalent iff their canonical forms are equal.


CanonNode = tuple[int, tuple]  # (own_items, children)


def canonicalize_spec(node: Any) -> CanonNode:
    """Convert one node of an expected `structure:` spec to canonical form.

    Node grammar:
      - int N        -> leaf with N items, 0 subscales
      - list [...]   -> composite with 0 own items and those entries as subscales
      - dict         -> {items: N, subscales: [...]} composite with N own items
    """
    if isinstance(node, bool):
        raise ValueError(f"structure spec: bool is not a valid node: {node!r}")
    if isinstance(node, int):
        return (node, ())
    if isinstance(node, list):
        children = tuple(sorted(canonicalize_spec(c) for c in node))
        return (0, children)
    if isinstance(node, dict):
        items = node.get("items", 0)
        subs = node.get("subscales", [])
        if not isinstance(items, int) or isinstance(items, bool):
            raise ValueError(f"structure spec: 'items' must be an int, got {items!r}")
        if not isinstance(subs, list):
            raise ValueError(f"structure spec: 'subscales' must be a list, got {subs!r}")
        children = tuple(sorted(canonicalize_spec(c) for c in subs))
        return (items, children)
    raise ValueError(f"structure spec: unsupported node {node!r} ({type(node).__name__})")


def canonicalize_expected(spec: Any) -> CanonNode:
    """Canonicalize the top-level `structure:` value (a list of top-level scales)."""
    if not isinstance(spec, list):
        raise ValueError(
            f"top-level structure: must be a list of top-level scales, got {type(spec).__name__}"
        )
    children = tuple(sorted(canonicalize_spec(c) for c in spec))
    return (0, children)


def canonicalize_scale(scale: Scale) -> CanonNode:
    own_items = len(scale.items) if not scale.subscales else 0
    children = tuple(sorted(canonicalize_scale(s) for s in scale.subscales))
    return (own_items, children)


def canonicalize_actual(survey: Survey) -> CanonNode:
    """Canonicalize the extracted Survey's scale tree the same way as the spec.

    A scale is treated as a 'leaf' (own_items = len(scale.items)) iff it has
    no subscales. Composite scales re-list their facets' items per the model
    docstring; counting those as own_items would double-count, so composites
    report own_items = 0. Match this rule when writing specs: use dict form
    only if a composite genuinely carries its own items separate from its
    subscales' items.
    """
    children = tuple(sorted(canonicalize_scale(s) for s in survey.scales))
    return (0, children)


def format_tree(node: CanonNode, indent: int = 0) -> str:
    """Human-readable rendering of a canonical tree."""
    own, children = node
    pad = "  " * indent
    if not children:
        return f"{pad}leaf({own})"
    body = "\n".join(format_tree(c, indent + 1) for c in children)
    return f"{pad}node(items={own}):\n{body}"
