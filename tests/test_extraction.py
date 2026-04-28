"""Three assertions per extraction:

1. Extraction produced a valid Survey.
2. Survey is in the expected language (defaults to English).
3. The scale tree matches the expected hierarchical structure (order- and
   name-insensitive, checked via canonical form).
"""
from __future__ import annotations

import pytest

from utils import (
    ExpectedCase,
    canonicalize_observed,
    canonicalize_expected,
    check_item,
    check_property,
    check_scale,
    collect_property,
    find_items_by_text,
    format_as_yaml_structure,
)


DEFAULT_LANGUAGE = "en"


def _get_result(case: ExpectedCase, run_index: int, get_extraction):
    return get_extraction(case.pdf, run_index)


def test_extraction_succeeded(case: ExpectedCase, run_index: int, get_extraction):
    result = _get_result(case, run_index, get_extraction)
    assert result.survey is not None, (
        f"extraction failed to produce a valid Survey: {result.validation_error}"
    )


def test_language(case: ExpectedCase, run_index: int, get_extraction):
    expected = case.spec.get("language") or DEFAULT_LANGUAGE
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check language: {result.validation_error}")
    assert result.survey.language == expected, (
        f"Survey.language={result.survey.language!r}, expected {expected!r}"
    )


def test_structure(case: ExpectedCase, run_index: int, get_extraction):
    spec = case.spec.get("structure")
    if spec is None:
        pytest.skip("no structure spec for this case")
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check structure: {result.validation_error}")

    expected = canonicalize_expected(spec)
    observed = canonicalize_observed(result.survey)
    assert observed == expected, (
        "scale tree does not match expected structure.\n"
        f"expected:\n{format_as_yaml_structure(expected)}\n"
        f"observed:\n{format_as_yaml_structure(observed)}"
    )


def test_property(case: ExpectedCase, run_index: int, property_check: dict, get_extraction):
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check property: {result.validation_error}")
    values = collect_property(result.survey, property_check["name"])
    ok, diag = check_property(values, property_check)
    assert ok, f"{property_check['name']}: {diag}"


def test_item(case: ExpectedCase, run_index: int, item_check: dict, get_extraction):
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check item: {result.validation_error}")
    matches = find_items_by_text(result.survey, item_check["text"])
    ok, diag = check_item(matches, item_check)
    assert ok, f"item[text~={item_check['text']!r}]: {diag}"


def test_scale(case: ExpectedCase, run_index: int, scale_check: dict, get_extraction):
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check scale: {result.validation_error}")
    ok, diag = check_scale(result.survey, scale_check)
    assert ok, f"scale[name={scale_check.get('scale_name')!r}]: {diag}"
