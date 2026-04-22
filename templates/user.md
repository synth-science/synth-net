The images show scanned pages of a questionnaire or survey. Extract the survey structure using the following schema and **strictly** adhere to the documentation. If no English translation of the content of the questionnaire or survey is supplied, translate the content before extracting it.

## Items belong inside scales

Items only exist in the context of a scale — there is **no separate top-level items list**. Every item must be listed inline inside the `items` array of the scale (or subscale) that scores it. A scale with no items is invalid — `items` must contain at least one entry.

If the same item is scored into more than one scale (e.g. a facet item that also rolls into a parent composite), list it inside **each** scale verbatim — use identical `item_text` in every occurrence so the validator can collapse them into a single logical item. Per-scale fields like `reverse_keyed` are kept independently for each occurrence.

### Minimal shape (illustrative — not a real instrument)

```json
{
  "scales": [
    {
      "scale_id": 1,
      "scale_name": "Extraversion",
      "construct_name": "Extraversion",
      "language": "en",
      "items": [
        {
          "item_text": "I see myself as someone who is talkative.",
          "response_format_id": 1,
          "language": "en",
          "reverse_keyed": false
        },
        {
          "item_text": "I see myself as someone who is reserved.",
          "response_format_id": 1,
          "language": "en",
          "reverse_keyed": true
        }
      ]
    }
  ]
}
```

Demographic / admin / clinical-context questions that are **not** scored into any scale belong in the top-level `auxiliary_items` array, not inside any scale.
