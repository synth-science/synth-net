"""Assertion tests for survey extraction.

Each test function covers one assertion category. All are parametrized over
(case, run_index) by conftest.pytest_generate_tests, so the default pytest
output shows N rows per PDF per assertion — the natural per-assertion
pass rate across stochastic runs.

Expected-value spec sections in the YAML fixture are all optional. A test
function is skipped (not failed) if the spec omits the relevant section.
"""
from __future__ import annotations

import pytest

from utils import (
    ExpectedCase,
    check_range,
    contains_any,
    detect_english_fraction,
    find_item_by_keywords,
    normalize,
    scales_containing_item,
    scales_matching_keywords,
    walk_scales,
)


def _get_result(case: ExpectedCase, run_index: int, get_extraction):
    return get_extraction(case.pdf, run_index)


# ---------------------------------------------------------------------------
# 1. Extraction succeeded (parsed a valid Survey)
# ---------------------------------------------------------------------------

def test_extract_succeeded(case: ExpectedCase, run_index: int, get_extraction):
    result = _get_result(case, run_index, get_extraction)
    assert result.survey is not None, (
        f"extraction failed to produce a valid Survey: {result.validation_error}"
    )


# ---------------------------------------------------------------------------
# 2. Language + langdetect
# ---------------------------------------------------------------------------

def test_lang(case: ExpectedCase, run_index: int, config, get_extraction):
    spec = case.spec.get("language")
    if not spec:
        pytest.skip("no language spec for this case")
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check language: {result.validation_error}")

    survey = result.survey
    target = spec.get("expect_translated_to")
    if target:
        assert survey.language == target, (
            f"Survey.language={survey.language!r}, expected {target!r}"
        )
        assert survey.is_translated is True, (
            "Survey.is_translated should be True for translated docs"
        )
        threshold = float(
            spec.get(
                "langdetect_threshold",
                (config.get("tests") or {}).get("langdetect_threshold", 0.8),
            )
        )
        item_texts = [i.item_text for i in survey.items]
        fraction = detect_english_fraction(item_texts)
        assert fraction >= threshold, (
            f"only {fraction:.2f} of item texts detected as English "
            f"(threshold {threshold:.2f})"
        )
    else:
        expected_lang = spec.get("source")
        if expected_lang:
            assert survey.language == expected_lang, (
                f"Survey.language={survey.language!r}, expected {expected_lang!r}"
            )


# ---------------------------------------------------------------------------
# 3. Counts (items, scales, aux items, response formats)
# ---------------------------------------------------------------------------

def test_counts(case: ExpectedCase, run_index: int, get_extraction):
    spec = case.spec.get("counts")
    if not spec:
        pytest.skip("no counts spec for this case")
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check counts: {result.validation_error}")

    survey = result.survey
    top_level_scales = len(survey.scales)
    all_scales = len(list(walk_scales(survey.scales)))

    checks = {
        "items": (len(survey.items), spec.get("items")),
        "scales_top_level": (top_level_scales, spec.get("scales_top_level")),
        "scales_total": (all_scales, spec.get("scales_total")),
        "auxiliary_items": (len(survey.auxiliary_items), spec.get("auxiliary_items")),
        "response_formats": (len(survey.response_formats), spec.get("response_formats")),
    }

    failures = []
    for name, (value, range_spec) in checks.items():
        if range_spec is None:
            continue
        ok, msg = check_range(value, range_spec)
        if not ok:
            failures.append(f"{name}: {msg}")
    assert not failures, "count mismatches: " + "; ".join(failures)


# ---------------------------------------------------------------------------
# 4. Scales: expected scales exist with scored-item counts in range
# ---------------------------------------------------------------------------

def test_scales(case: ExpectedCase, run_index: int, get_extraction):
    scale_specs = case.spec.get("scales")
    if not scale_specs:
        pytest.skip("no scales spec for this case")
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check scales: {result.validation_error}")

    survey = result.survey
    failures = []
    for spec in scale_specs:
        keywords = spec.get("name_keywords_any_of") or []
        matches = scales_matching_keywords(survey, keywords)
        if not matches:
            names = ", ".join(s.scale_name for s in walk_scales(survey.scales))
            failures.append(
                f"no scale matched {keywords!r}. present scales: [{names}]"
            )
            continue
        count_spec = spec.get("scored_items_count")
        if count_spec is not None:
            # Prefer the match with the most scored items (handles the composite case).
            best = max(matches, key=lambda s: len(s.scored_items))
            ok, msg = check_range(len(best.scored_items), count_spec)
            if not ok:
                failures.append(f"scale {best.scale_name!r} scored_items: {msg}")
    assert not failures, "scale mismatches: " + "; ".join(failures)


# ---------------------------------------------------------------------------
# 5. Items found — each expected item uniquely matches one extracted item
# ---------------------------------------------------------------------------

def test_items_found(case: ExpectedCase, run_index: int, get_extraction):
    item_specs = case.spec.get("items")
    if not item_specs:
        pytest.skip("no items spec for this case")
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check items: {result.validation_error}")

    survey = result.survey
    failures = []
    for spec in item_specs:
        kws = spec.get("find_by_keywords") or []
        _, diag = find_item_by_keywords(survey, kws)
        if diag:
            failures.append(diag)
    assert not failures, "item match failures: " + " | ".join(failures)


# ---------------------------------------------------------------------------
# 6. Item -> scale linkage
# ---------------------------------------------------------------------------

def test_item_scale_links(case: ExpectedCase, run_index: int, get_extraction):
    item_specs = case.spec.get("items")
    if not item_specs or not any("in_scale_keywords_any_of" in s for s in item_specs):
        pytest.skip("no item-scale linkage spec for this case")
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check linkage: {result.validation_error}")

    survey = result.survey
    failures = []
    for spec in item_specs:
        expected_scale_kws = spec.get("in_scale_keywords_any_of")
        if not expected_scale_kws:
            continue
        kws = spec.get("find_by_keywords") or []
        item, diag = find_item_by_keywords(survey, kws)
        if item is None:
            failures.append(f"[unmatched item {kws!r}] {diag}")
            continue
        owning = scales_containing_item(survey, item.item_id)
        if not owning:
            failures.append(
                f"item #{item.item_id} ({item.item_text[:60]!r}) is not scored by any scale"
            )
            continue
        if not any(contains_any(s.scale_name, expected_scale_kws) for s in owning):
            owners = ", ".join(s.scale_name for s in owning)
            failures.append(
                f"item #{item.item_id} is scored by [{owners}], "
                f"expected a scale matching {expected_scale_kws!r}"
            )
    assert not failures, "item-scale link failures: " + " | ".join(failures)


# ---------------------------------------------------------------------------
# 7. Reverse-keyed flags
# ---------------------------------------------------------------------------

def test_reverse_keyed(case: ExpectedCase, run_index: int, get_extraction):
    item_specs = case.spec.get("items")
    if not item_specs or not any("reverse_keyed" in s for s in item_specs):
        pytest.skip("no reverse-keyed spec for this case")
    result = _get_result(case, run_index, get_extraction)
    if result.survey is None:
        pytest.fail(f"extraction failed, cannot check reverse-keying: {result.validation_error}")

    survey = result.survey
    failures = []
    for spec in item_specs:
        if "reverse_keyed" not in spec:
            continue
        expected = bool(spec["reverse_keyed"])
        kws = spec.get("find_by_keywords") or []
        item, diag = find_item_by_keywords(survey, kws)
        if item is None:
            failures.append(f"[unmatched item {kws!r}] {diag}")
            continue
        # Collect every ScoredItem pointing at this item across the scale tree.
        scored_refs = [
            si
            for s in walk_scales(survey.scales)
            for si in s.scored_items
            if si.item_id == item.item_id
        ]
        if not scored_refs:
            failures.append(
                f"item #{item.item_id} has no scored_items entries; "
                f"cannot verify reverse_keyed"
            )
            continue
        # Pass if any scored_items entry matches expected — a reverse-keyed item
        # is usually reverse-keyed in every scale that scores it, but the spec
        # is satisfied as soon as one match is found.
        if not any(si.reverse_keyed == expected for si in scored_refs):
            observed = [si.reverse_keyed for si in scored_refs]
            failures.append(
                f"item #{item.item_id} ({normalize(item.item_text)[:50]!r}): "
                f"expected reverse_keyed={expected}, observed={observed}"
            )
    assert not failures, "reverse-key failures: " + " | ".join(failures)
