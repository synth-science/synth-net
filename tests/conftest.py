"""Pytest configuration for the extraction test harness.

Each test case in tests/expected/<pdf_stem>.yaml is parametrized over
`runs_per_case` runs so the default pytest report naturally shows a
per-assertion pass rate across stochastic extractions.

A session-scoped cache holds one `ExtractionResult` per (pdf, run_index)
so the seven assertion tests (language, counts, scales, ...) reuse the
same extraction within a single pytest session.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml

# Make main.py and models.py importable without installing the project.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import extract_survey, load_config, ExtractionResult  # noqa: E402
from utils import ExpectedCase  # noqa: E402


def _load_expected_cases(config: dict) -> list[ExpectedCase]:
    expected_dir = REPO_ROOT / (config.get("tests") or {}).get("expected_dir", "tests/expected")
    cases: list[ExpectedCase] = []
    if not expected_dir.exists():
        return cases
    for path in sorted(expected_dir.glob("*.yaml")):
        with open(path) as f:
            spec = yaml.safe_load(f)
        if not spec or "pdf" not in spec:
            continue
        cases.append(ExpectedCase(pdf=spec["pdf"], path=path, spec=spec))
    return cases


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(str(REPO_ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def extraction_cache() -> dict[tuple[str, int], ExtractionResult]:
    return {}


@pytest.fixture
def get_extraction(extraction_cache, config):
    """Fixture returning a callable (pdf, run_index) -> ExtractionResult.

    Memoizes per (pdf, run_index) so the seven assertion tests for one
    (pdf, run_index) combination share a single ollama call.
    """
    def _fetch(pdf: str, run_index: int) -> ExtractionResult:
        key = (pdf, run_index)
        if key not in extraction_cache:
            pdf_path = Path(config["input_dir"]) / pdf
            extraction_cache[key] = extract_survey(pdf_path, config)
        return extraction_cache[key]
    return _fetch


# --------------------------------------------------------------------------
# Parametrization: discover expected files + fan out across N runs.
# --------------------------------------------------------------------------

# Tests opt into parametrization by accepting `case` and `run_index` params.

def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames and "run_index" in metafunc.fixturenames:
        config = load_config(str(REPO_ROOT / "config.yaml"))
        runs = int((config.get("tests") or {}).get("runs_per_case", 1))
        cases = _load_expected_cases(config)
        params = [(c, r) for c in cases for r in range(runs)]
        ids = [f"{c.pdf}-run{r}" for c, r in params]
        metafunc.parametrize(("case", "run_index"), params, ids=ids)


# --------------------------------------------------------------------------
# Per-assertion pass-rate matrix printed at the end of the session.
# --------------------------------------------------------------------------

# (pdf, assertion_group) -> [passed_bool, ...]
_results: dict[tuple[str, str], list[bool]] = defaultdict(list)


# Assertion groups are identified by a substring of the test function name.
# Keep aligned with test_extraction.py.
_ASSERTION_GROUPS = [
    ("succeeded", "extract"),
    ("language", "lang"),
    ("counts", "counts"),
    ("scales", "scales"),
    ("items_found", "items"),
    ("item_scale", "links"),
    ("reverse_keyed", "rev-key"),
]


def _group_for(nodeid: str) -> str | None:
    for needle, label in _ASSERTION_GROUPS:
        if needle in nodeid:
            return label
    return None


def _pdf_for(nodeid: str) -> str | None:
    # nodeids look like: tests/test_extraction.py::test_counts[999901837_full_001.pdf-run0]
    if "[" not in nodeid:
        return None
    bracket = nodeid.split("[", 1)[1].rstrip("]")
    if "-run" not in bracket:
        return bracket
    return bracket.rsplit("-run", 1)[0]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or report.skipped:
        return
    pdf = _pdf_for(report.nodeid)
    group = _group_for(report.nodeid)
    if pdf and group:
        _results[(pdf, group)].append(report.passed)


def pytest_terminal_summary(terminalreporter, exitstatus, config: Any):
    if not _results:
        return
    pdfs = sorted({pdf for pdf, _ in _results})
    groups = [label for _, label in _ASSERTION_GROUPS]

    tr = terminalreporter
    tr.write_sep("=", "extraction robustness matrix")
    header = f"{'pdf':<40}" + "".join(f"{g:>10}" for g in groups)
    tr.write_line(header)
    for pdf in pdfs:
        row = f"{pdf:<40}"
        for g in groups:
            outcomes = _results.get((pdf, g), [])
            if not outcomes:
                cell = "-"
            else:
                passed = sum(outcomes)
                total = len(outcomes)
                cell = f"{passed}/{total}"
            row += f"{cell:>10}"
        tr.write_line(row)
