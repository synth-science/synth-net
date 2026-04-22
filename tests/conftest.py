"""Pytest configuration for the extraction test harness.

Each `tests/expected/<pdf_stem>.yaml` is one parametrized case, fanned out
over `runs_per_case` (for stochastic retries). Cases with
`exclude_from_testing: true` are silently skipped during discovery.

A session-scoped cache holds one `ExtractionResult` per (pdf, run_index)
so the three assertion tests reuse the same extraction. At the end of the
session two summary tables are printed and also written to
`logs/test-report_<timestamp>.log`.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import extract_survey, load_config, ExtractionResult  # noqa: E402
from utils import ExpectedCase  # noqa: E402


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

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
        if spec.get("exclude_from_testing"):
            continue
        cases.append(ExpectedCase(pdf=spec["pdf"], path=path, spec=spec))
    return cases


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(str(REPO_ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def extraction_cache() -> dict[tuple[str, int], ExtractionResult]:
    return {}


@pytest.fixture
def get_extraction(extraction_cache, config):
    """Callable (pdf, run_index) -> ExtractionResult, memoized per (pdf, run_index)."""
    def _fetch(pdf: str, run_index: int) -> ExtractionResult:
        key = (pdf, run_index)
        if key not in extraction_cache:
            pdf_path = Path(config["input_dir"]) / pdf
            extraction_cache[key] = extract_survey(pdf_path, config)
        return extraction_cache[key]
    return _fetch


# --------------------------------------------------------------------------
# Parametrization
# --------------------------------------------------------------------------

_PROPERTY_OPS = ("equals", "not_null", "contains", "in")


def _check_id(check: dict) -> str:
    name = check.get("name", "?")
    op = next((k for k in _PROPERTY_OPS if k in check), None)
    if op is None:
        return f"{name}:no-op"
    val = check[op]
    if op == "not_null":
        return f"{name}:not_null"
    val_repr = str(val)
    if len(val_repr) > 20:
        val_repr = val_repr[:17] + "..."
    return f"{name}:{op}={val_repr}"


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames or "run_index" not in metafunc.fixturenames:
        return
    config = load_config(str(REPO_ROOT / "config.yaml"))
    runs = int((config.get("tests") or {}).get("runs_per_case", 1))
    cases = _load_expected_cases(config)

    if "property_check" in metafunc.fixturenames:
        params, ids = [], []
        for c in cases:
            for check in (c.spec.get("properties") or []):
                for r in range(runs):
                    params.append((c, r, check))
                    ids.append(f"{c.pdf}-run{r}-{_check_id(check)}")
        metafunc.parametrize(("case", "run_index", "property_check"), params, ids=ids)
    else:
        params = [(c, r) for c in cases for r in range(runs)]
        ids = [f"{c.pdf}-run{r}" for c, r in params]
        metafunc.parametrize(("case", "run_index"), params, ids=ids)


# --------------------------------------------------------------------------
# Per-run reporting
# --------------------------------------------------------------------------

# Assertion label derived from the test function name suffix (after "test_").
# Kept in sync with tests/test_extraction.py.
_ASSERTION_LABELS = {
    "test_extraction_succeeded": "succeeded",
    "test_language": "language",
    "test_structure": "structure",
    "test_property": "property",
}

# Collected results: list of (pdf, assertion, passed, diagnostic)
_records: list[tuple[str, str, bool, str]] = []


def _parse_nodeid(nodeid: str) -> tuple[str | None, str | None]:
    # nodeid: tests/test_extraction.py::test_structure[999967007_full_001.pdf-run0]
    func_part, _, param_part = nodeid.partition("[")
    func_name = func_part.rsplit("::", 1)[-1]
    assertion = _ASSERTION_LABELS.get(func_name)
    pdf = None
    if param_part:
        bracket = param_part.rstrip("]")
        pdf = bracket.rsplit("-run", 1)[0] if "-run" in bracket else bracket
    return pdf, assertion


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    if report.skipped:
        return
    pdf, assertion = _parse_nodeid(report.nodeid)
    if not pdf or not assertion:
        return
    diagnostic = ""
    if not report.passed:
        diagnostic = (str(report.longrepr).splitlines() or [""])[-1].strip()
    _records.append((pdf, assertion, bool(report.passed), diagnostic))


def _render_by_document() -> list[str]:
    by_doc: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for pdf, assertion, passed, _ in _records:
        by_doc[pdf].append((assertion, passed))

    lines = []
    header = f"{'document':<42}{'passed':>10}{'total':>8}  missed"
    lines.append(header)
    lines.append("-" * len(header))
    for pdf in sorted(by_doc):
        entries = by_doc[pdf]
        total = len(entries)
        passed = sum(1 for _, p in entries if p)
        missed = ", ".join(sorted(a for a, p in entries if not p)) or "-"
        lines.append(f"{pdf:<42}{f'{passed}/{total}':>10}{total:>8}  {missed}")
    return lines


def _render_by_assertion() -> list[str]:
    by_assertion: dict[str, list[bool]] = defaultdict(list)
    for _, assertion, passed, _ in _records:
        by_assertion[assertion].append(passed)

    lines = []
    header = f"{'assertion':<14}{'passed':>10}{'total':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    # Preserve the declared order from _ASSERTION_LABELS.
    ordered = [lbl for lbl in _ASSERTION_LABELS.values() if lbl in by_assertion]
    for assertion in ordered:
        outcomes = by_assertion[assertion]
        total = len(outcomes)
        passed = sum(outcomes)
        lines.append(f"{assertion:<14}{f'{passed}/{total}':>10}{total:>8}")
    return lines


def _render_failures() -> list[str]:
    fails = [(pdf, a, d) for pdf, a, p, d in _records if not p]
    if not fails:
        return ["(no failures)"]
    return [f"{pdf} :: {a}  —  {d or '(no diagnostic)'}" for pdf, a, d in sorted(fails)]


def _build_report(timestamp: str) -> str:
    parts = [f"test-report  {timestamp}", ""]
    parts.append("## By document")
    parts.extend(_render_by_document())
    parts.append("")
    parts.append("## By assertion")
    parts.extend(_render_by_assertion())
    parts.append("")
    parts.append("## Failures")
    parts.extend(_render_failures())
    parts.append("")
    return "\n".join(parts)


def pytest_terminal_summary(terminalreporter, exitstatus, config: Any):
    if not _records:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report = _build_report(timestamp)

    tr = terminalreporter
    tr.write_sep("=", "extraction test report")
    for line in report.splitlines():
        tr.write_line(line)

    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"test-report_{timestamp}.log"
    log_path.write_text(report)
    tr.write_line(f"report written to {log_path}")
