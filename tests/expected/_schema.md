# Expected-value fixtures

Each `.yaml` file is one test case. The filename is cosmetic; `pdf:` is
authoritative and must match a file in `config.input_dir`.

Three assertions run per case: extraction success, language, structure.

## Shape

```yaml
pdf: 999967007_full_001.pdf        # required
exclude_from_testing: false        # optional; true skips the case

notes: |                           # optional; free text, ignored by tests
  Anything useful about the case.

language: en                       # optional; defaults to "en". Survey.language must equal this.

structure:                         # optional; if omitted, test_structure is skipped
  - 3                              # top scale A: leaf with 3 items
  - [4, 5]                         # top scale B: 2 subscales (leaves of 4, 5 items)
```

## Structure DSL

At any node position the spec is one of:

| Form | Meaning |
|------|---------|
| `N` (int) | Leaf scale with `N` items, no subscales. |
| `[...]` (list) | Composite scale with `0` own items and those entries as subscales. |
| `{items: N, subscales: [...]}` | Composite with `N` own items plus those subscales. |

The root of `structure:` is the list of top-level scales.

### Example

The shape "two top-level scales; the first has 3 items; the second has two
subscales of 4 and 5 items":

```yaml
structure: [3, [4, 5]]
```

### Matching

Comparison is via canonical form: both expected and actual trees are
reduced to sorted nested tuples `(own_items, sorted(children))`. Sibling
order doesn't matter; scale names are never consulted. Any mismatch
(own-item counts or child shapes) fails the assertion with a pretty-printed
expected/actual diff.

## Disabling a case

Set `exclude_from_testing: true` at the top of the file. The harness skips
the case during discovery — it won't appear in any test output or report.

## Running

- `poetry run pytest tests/ -v` — runs all non-excluded cases × `tests.runs_per_case`.
- Two summary tables print after the run (by-document and by-assertion).
- A copy of the report is written to `logs/test-report_<timestamp>.log`.
