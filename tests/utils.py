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


def _canon_to_spec(node: CanonNode) -> Any:
    own_items, children = node
    if not children:
        return own_items
    child_specs = [_canon_to_spec(c) for c in children]
    if own_items == 0:
        return child_specs
    return {"items": own_items, "subscales": child_specs}


def _spec_to_flow(spec: Any) -> str:
    if isinstance(spec, int):
        return str(spec)
    if isinstance(spec, list):
        return "[" + ", ".join(_spec_to_flow(x) for x in spec) + "]"
    if isinstance(spec, dict):
        subs = "[" + ", ".join(_spec_to_flow(x) for x in spec.get("subscales", [])) + "]"
        return "{items: " + str(spec["items"]) + ", subscales: " + subs + "}"
    return repr(spec)


def format_as_yaml_structure(node: CanonNode) -> str:
    """Render a CanonNode in the same block-outer/flow-inner YAML syntax used in .yaml fixtures."""
    _, children = node
    return "\n".join("- " + _spec_to_flow(_canon_to_spec(c)) for c in children)


# ---------------------------------------------------------------------------
# Property-level assertions
# ---------------------------------------------------------------------------
#
# Walk the dumped survey dict and pick up every value bound to a given key
# anywhere in the hierarchy (Survey / Scale / subscale / ScoredItem /
# ResponseFormat / AuxiliaryItem). Any-of matching: the assertion passes if
# any collected value satisfies the operator.


PROPERTY_OPS = ("equals", "not_null", "contains", "in")


def collect_property(survey: Survey, name: str) -> list:
    out: list = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                out.append(obj[name])
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    visit(survey.model_dump())
    return out


def check_property(values: list, spec: dict) -> tuple[bool, str]:
    op = next((k for k in PROPERTY_OPS if k in spec), None)
    if op is None:
        return False, f"no operator (expected one of {PROPERTY_OPS})"
    expected = spec[op]
    if op == "not_null":
        ok = any(v not in (None, "", [], {}) for v in values)
    elif op == "equals":
        ok = any(v == expected for v in values)
    elif op == "contains":
        ok = any(isinstance(v, str) and expected in v for v in values)
    elif op == "in":
        ok = any(v in expected for v in values)
    else:
        ok = False
    return ok, f"{op}={expected!r}; found {values!r}"


# ---------------------------------------------------------------------------
# Item-level assertions
# ---------------------------------------------------------------------------
#
# Pick ScoredItems out of the survey by an item_text substring, then assert
# equality on any remaining fields declared in the spec (e.g. reverse_keyed).
# Any-of matching: at least one matching item must satisfy all field checks.


def find_items_by_text(survey: Survey, text: str) -> list[dict]:
    """Return every ScoredItem dict in the dumped survey whose item_text contains `text`."""
    needle = normalize(text)
    out: list[dict] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            if "item_text" in obj and "reverse_keyed" in obj:
                if needle in normalize(obj.get("item_text") or ""):
                    out.append(obj)
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    visit(survey.model_dump())
    return out


def check_item(matches: list[dict], spec: dict) -> tuple[bool, str]:
    fields = {k: v for k, v in spec.items() if k != "text"}
    if not fields:
        return False, f"no field assertions declared (need at least one besides 'text')"
    if not matches:
        return False, f"no item contains text {spec['text']!r}"
    ok = any(all(m.get(f) == v for f, v in fields.items()) for m in matches)
    if not ok:
        seen = [{k: m.get(k) for k in fields} for m in matches]
        return False, (
            f"{len(matches)} item(s) matched text {spec['text']!r}; "
            f"none satisfy {fields}; got {seen}"
        )
    return True, f"{fields} satisfied by one of {len(matches)} match(es)"
