The images show scanned pages of a questionnaire or survey. Extract the survey structure using the following schema and **strictly** adhere to the documentation. 

**Important**: Also consider these rules:
- If no English translation of the content of the questionnaire or survey is supplied, translate the content before extracting it. 
- Don't put item numbers into the `item_text´ field.
- Also exctract drafted items which did not make it to the final version of a questionnaire, if available.
- Copy all `item_text` item and `scale_names` verbatim from the source - do not paraphrase, correct spelling, fix grammar, or alter wording in any way, even if the text appears to contain typos or grammatically incorrect sentences
- After assigning `reverse_keyed` for each item, reason about whether the overall pattern of reverse-keyed items makes sense given their phrasing. If you are strongly confident that one or more items are incorrectly labeled, correct the `reverse_keyed` value and set `keying_corrected` to true for those items only.
- If a document mentions or describes multiple scales but does not clearly assign items to them, use reasoning to determine the most likely assignment per item. Set `is_inferred` to true only when the mapping required active inference; leave `is_inferred` false if the assignment is unambiguous.