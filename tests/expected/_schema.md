# Expected-value fixtures

Each `.yaml` file in this directory is one test case. The filename is cosmetic;
the `pdf:` field is authoritative and must match a file in `config.input_dir`.

All sections other than `pdf:` are **optional**. If a section is absent, the
corresponding assertion test is skipped for that case. This lets you add
cases incrementally — start with counts, add scales/items later.

## Full shape

```yaml
pdf: 999979446_full_001.pdf
notes: "Free-text memo. Not consumed by tests."

language:
  source: it                          # ISO 639-1 of the source document
  expect_translated_to: en            # omit if no translation is expected
  langdetect_threshold: 0.8           # optional per-case override

counts:
  # Each value is a range spec: {min: N}, {max: N}, {min, max}, or {exact: N}.
  items: {min: 20, max: 24}
  scales_top_level: {exact: 4}
  scales_total: {min: 4, max: 8}      # includes nested subscales
  auxiliary_items: {max: 3}
  response_formats: {min: 1, max: 2}

scales:
  - name_keywords_any_of: ["extraversion", "estroversione"]
    scored_items_count: {min: 4, max: 8}
  - name_keywords_any_of: ["neuroticism"]
    # scored_items_count is optional; omit to assert only existence.

items:
  - find_by_keywords: ["talkative"]
    in_scale_keywords_any_of: ["extraversion"]
    reverse_keyed: false
  - find_by_keywords: ["reserved", "quiet"]
    in_scale_keywords_any_of: ["extraversion"]
    reverse_keyed: true
```

## Matching rules

- `find_by_keywords`: **all** keywords must appear as substrings in
  `item_text` after lowercase + whitespace-normalization. The match must be
  unique — "no match" and "ambiguous match" both fail with a diagnostic
  listing candidate items.
- `in_scale_keywords_any_of`: after finding the item, collect every scale
  (at any depth) whose `scored_items` references the item's `item_id` and
  assert **at least one** scale name contains one of the keywords.
- `name_keywords_any_of` (for `scales:`): any scale in the tree whose name
  contains one of the keywords satisfies the entry.
- `reverse_keyed`: after finding the item, collect its `ScoredItem`
  entries across all scales; passes if any entry matches the expected flag.
  Fails if the item has no `ScoredItem` entries at all (cannot verify).

Keep keyword lists short (2–3 items). They should uniquely identify the item
by content, not test transcription fidelity — exact wording varies across
translations and runs.

## Running

- `poetry run pytest tests/ -v` — runs all cases × `tests.runs_per_case`
  runs. The per-assertion pass rate matrix prints in the terminal summary.
- Set `tests.runs_per_case` in `config.yaml` to control repetition.
- Set `tests.ollama_temperature` to encourage output variance across runs.
