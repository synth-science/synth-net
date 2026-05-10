# Expected-value fixtures

Each `.yaml` file is one test case. The filename is cosmetic; `pdf:` is
authoritative and must match a file in `config.input_dir`.

Per case: three baseline assertions (extraction success, language, structure)
plus one additional assertion per entry in the optional `properties:` list.

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

properties:                        # optional; one assertion per entry
  - name: items_prefix
    not_null: true
  - name: report_type
    equals: "other-report"

items:                             # optional; one assertion per entry
  - text: "I worry about being alone"
    reverse_keyed: true
  - text: "I enjoy meeting new people"
    reverse_keyed: false
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

Comparison is via canonical form: both expected and observed trees are
reduced to sorted nested tuples `(own_items, sorted(children))`. Sibling
order doesn't matter; scale names are never consulted. Any mismatch
(own-item counts or child shapes) fails the assertion with a pretty-printed
expected/observed diff.

## Property assertions

Use `properties:` to assert on individual fields (e.g. `items_prefix`,
`report_type`, `language`, `is_translated`) without caring where in the
hierarchy they live. The harness walks the entire extracted survey
(Survey → scales → subscales → items → response_formats → auxiliary_items)
and collects every value bound to `name`. The check passes if **any**
collected value satisfies the operator (any-of matching).

Each entry is `{name, <one operator>}`. Multiple checks on the same
property name are allowed — each becomes its own pytest node.

### Operators

| Operator | Passes when... | Example |
|----------|----------------|---------|
| `equals: V` | some value `== V` | `equals: "other-report"` |
| `not_null: true` | some value is non-empty (not `None`/`""`/`[]`/`{}`) | `not_null: true` |
| `contains: S` | some string value contains substring `S` | `contains: "rate"` |
| `in: [...]` | some value is in the allowed list | `in: ["en", "de"]` |

### Example

```yaml
properties:
  - name: items_prefix          # any node has items_prefix set
    not_null: true
  - name: items_prefix          # ...and at least one mentions "rate"
    contains: "rate"
  - name: report_type           # any scale's report_type matches
    equals: "other-report"
  - name: language              # survey/scale/item language is en or de
    in: ["en", "de"]
```

### Failure diagnostic

On failure the assertion prints the operator, the expected value, and the
list of values actually collected from the hierarchy — so you can tell
whether the property was missing entirely or just had the wrong value.

## Item assertions

Use `items:` to pin claims to a specific ScoredItem identified by an
`item_text` substring. Each entry is one pytest node.

```yaml
items:
  - text: "I worry about being alone"   # substring, case-insensitive, whitespace-normalized
    reverse_keyed: true
  - text: "I enjoy meeting new people"
    reverse_keyed: false
```

- `text`: required. Normalized (lowercase + collapsed whitespace) substring
  match against `item_text`. AuxiliaryItems are not considered.
- Every other key is an equality check against that field on the matched
  item (e.g. `reverse_keyed`, `has_image`, `language`, `is_translated`).
- **Any-of matching**: the assertion passes if at least one item whose text
  matches satisfies **all** declared field checks. Since `reverse_keyed` is
  per-scale, the same item_text may appear with different values across
  scales; use a longer substring to disambiguate if that matters.

### Failure diagnostic

Two distinct failure modes, each with its own message:

- **No item found**: `no item contains text "<text>"`.
- **Fields mismatch**: `N item(s) matched text "<text>"; none satisfy {...};
  got [{...}, {...}]` — the collected field values are printed so you can
  see whether the extractor put the right item in the wrong state or the
  wrong item got the text match.

## Disabling a case

Set `exclude_from_testing: false` at the top of the file. The harness skips
the case during discovery — it won't appear in any test output or report.

## Running

- `poetry run pytest tests/ -v` — runs all non-excluded cases × `tests.runs_per_case`.
- Two summary tables print after the run (by-document and by-assertion).
- A copy of the report is written to `logs/test-report_<timestamp>.log`.
