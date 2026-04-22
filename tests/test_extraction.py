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
    canonicalize_actual,
    canonicalize_expected,
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
    actual = canonicalize_actual(result.survey)
    assert actual == expected, (
        "scale tree does not match expected structure.\n"
        f"expected:\n{format_as_yaml_structure(expected)}\n"
        f"actual:\n{format_as_yaml_structure(actual)}"
    )
