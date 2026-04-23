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

import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

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


_extraction_cache_ref: dict[tuple[str, int], ExtractionResult] | None = None


@pytest.fixture(scope="session")
def extraction_cache() -> dict[tuple[str, int], ExtractionResult]:
    global _extraction_cache_ref
    cache: dict[tuple[str, int], ExtractionResult] = {}
    _extraction_cache_ref = cache
    return cache


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

# Kept in sync with tests/test_extraction.py.
_ASSERTION_LABELS = {
    "test_extraction_succeeded": "succeeded",
    "test_language": "language",
    "test_structure": "structure",
    "test_property": "property",
}

_records: list[tuple[str, int, str, bool, str]] = []
_extractions: list[tuple[str, int, ExtractionResult]] = []
_seen_extractions: set[tuple[str, int]] = set()


def _parse_nodeid(nodeid: str) -> tuple[str | None, int, str | None]:

    func_part, _, param_part = nodeid.partition("[")
    func_name = func_part.rsplit("::", 1)[-1]
    assertion = _ASSERTION_LABELS.get(func_name)
    pdf = None
    run_index = 0
    if param_part:
        bracket = param_part.rstrip("]")
        if "-run" in bracket:
            pdf, run_str = bracket.rsplit("-run", 1)
            try:
                run_index = int(run_str.split("-")[0])
            except ValueError:
                run_index = 0
        else:
            pdf = bracket
    return pdf, run_index, assertion


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or report.skipped:
        return
    pdf, run_index, assertion = _parse_nodeid(report.nodeid)
    if not pdf or not assertion:
        return

    diagnostic = ""
    if not report.passed:
        if hasattr(report.longrepr, "reprcrash"):
            diagnostic = report.longrepr.reprcrash.message
        else:
            diagnostic = (str(report.longrepr).splitlines() or [""])[-1].strip()

    _records.append((pdf, run_index, assertion, bool(report.passed), diagnostic))

    if _extraction_cache_ref and "case" in item.funcargs:
        case = item.funcargs["case"]
        run_index = item.funcargs.get("run_index", 0)
        key = (case.pdf, run_index)
        if key not in _seen_extractions:
            result = _extraction_cache_ref.get(key)
            if result is not None:
                _seen_extractions.add(key)
                _extractions.append((case.pdf, run_index, result))


def _render_by_document() -> list[str]:
    by_doc: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for pdf, _run, assertion, passed, _ in _records:
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
    for _, _run, assertion, passed, _ in _records:
        by_assertion[assertion].append(passed)

    lines = []
    header = f"{'assertion':<14}{'passed':>10}{'total':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    
    ordered = [lbl for lbl in _ASSERTION_LABELS.values() if lbl in by_assertion]
    for assertion in ordered:
        outcomes = by_assertion[assertion]
        total = len(outcomes)
        passed = sum(outcomes)
        lines.append(f"{assertion:<14}{f'{passed}/{total}':>10}{total:>8}")
    return lines


def _filter_structure_diag(diag: str) -> str:
    """Strip pytest assertion-rewriting noise; keep only the YAML expected/actual block."""
    diag = _ANSI_RE.sub("", diag)
    lines = diag.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("expected:")), None)
    if start is None:
        marker = "scale tree does not match expected structure."
        idx = diag.find(marker)
        return diag[idx + len(marker):].lstrip("\n") if idx != -1 else diag
    end = next(
        (i for i, ln in enumerate(lines[start:], start) if ln.strip().startswith("assert ")),
        len(lines),
    )
    return "\n".join(lines[start:end]).rstrip()


def _render_failures() -> list[str]:
    fails = [(pdf, run, a, d) for pdf, run, a, p, d in _records if not p]
    if not fails:
        return ["(no failures)"]
    lines = []
    for pdf, run_index, assertion, diag in sorted(fails):
        lines.append(f"{pdf} :: run{run_index} :: {assertion}")
        if diag:
            if assertion == "structure":
                diag = _filter_structure_diag(diag)
            for diag_line in diag.splitlines():
                lines.append(f"  {diag_line}")
        else:
            lines.append("  (no diagnostic)")
        lines.append("")
    return lines


def _render_extracted_surveys() -> list[str]:
    if not _extractions:
        return ["(no extractions recorded)"]
    lines = []
    for pdf, run_index, result in _extractions:
        lines.append(f"### {pdf} :: run{run_index}")
        if result.survey is None:
            lines.append(f"  extraction failed: {result.validation_error}")
        else:
            data = result.survey.model_dump(exclude={"thinking"})
            dump = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
            lines.extend(f"  {line}" for line in dump.rstrip().splitlines())
        lines.append("")
    return lines


def _build_report(timestamp: str, include_surveys: bool = False) -> str:
    parts = [f"test-report  {timestamp}", ""]
    parts.append("## By document")
    parts.extend(_render_by_document())
    parts.append("")
    parts.append("## By assertion")
    parts.extend(_render_by_assertion())
    parts.append("")
    parts.append("## Failures")
    parts.extend(_render_failures())
    if include_surveys:
        parts.append("## Extracted surveys")
        parts.extend(_render_extracted_surveys())
    return "\n".join(parts)


def pytest_terminal_summary(terminalreporter, exitstatus, config: Any):
    if not _records:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    tr = terminalreporter
    tr.write_sep("=", "extraction test report")
    for line in _build_report(timestamp).splitlines():
        tr.write_line(line)

    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"test-report_{timestamp}.log"
    log_path.write_text(_build_report(timestamp, include_surveys=True))
    tr.write_line(f"report written to {log_path}")
