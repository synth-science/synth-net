TO DO's
- Improve models.py 🕐
- Loading modely.py as prompt should ignore `thinking` and `@model_validator`'s to save context
- Implement re-prompting
    - When error occurs, re-prompt telling model to fix error produced by validator
    - When same error occurs X times, fail extraction and move on
- Implement regex-extraction
- Implement Unit test
    - List filenames that should be tested and how many iterations (for robustness).
    - Testing different extraction challenges, e.g. translation, scale linkage, etc.
- write distributed computation logic

✅ Positive Patterns
- No errors in item count
- No hallucinated items

- REGEX:
    - PsycTEST Citation
    - Instrument Type
    - Test Format
    - Source (extract original DOI)
        Sometimes in conflict with `Original Publication` see `999903325_full_001.pdf`
    - Permissions

# Unit Test
## Code Issues:
- If Schema validation fails -> reprompt
!! ADD INSTRUMENT WITH ITEMS THAT HAVE CONTEXT

## Schema issues:
- add "version" property on survey 🕐
- add "abbrevation" property on survey 🕐
- inconsistent keys `response_format_id` (in `999901837_full_001.pdf`) vs `format_id` (in `999958013_full_002.pdf`)
    - My mistake:  `response_format_id` is nested under `items` wheras `format_id` is a property of `response_formats`
- consider adding field for "comments" in items, see `999967948_full_001.pdf`: "_What are the main things that you want to accomplish? (FOCUS ON MAIN 3)_"
    - for item comments, interviewer reads, etc.
- We need a way to distinguish all atypical from typical, well-behaved instruments
- consider adding "score" for response options, see `999943323_full_001`, where `{'label': 'Yes', 'value': 1}` while document suggests that "_yes_" and "_no_" are scored with _8_ and _0_ points each.
- Add `has_image` or similar property, ideally with an image description to sink image content as in `999915498_full_001.pdf`. 🕐
- Item context 🕐
- self-rating / other-rating 🕐


## Pass
- `999901837_full_001.pdf`: 
    ✅ Single scale, reverse coded items noted below items

## Hard Fails
- `999979446_full_001.pdf`: 
    ✅ poor resolution, difficult to transcribe
    ❌ italian, needs translation 🕐
- `999903325_full_001.pdf`:
    ❌ items need to be linked to scales via `["scales"]["items"]`
    'items':
- `999967948_full_001.pdf`:
    ❌ Varying response options in this questionnaire; model incorrectly identified questions in 11 as rating scales instead of single choice.
        - Messed up response formats in general: Questionnaire contains too many. Consider explicitly listing avail. response formats in pydantic documentation. Also clearly state that rating scales cannot have more than 10 points (?)
    ❌ Only linked first item to scale "Personal meaning of Quality of Life".
- `999980376_full_001.pdf`:
    ❌ Missing item to scale linkage
    ❌ Assumed scales despite lack of mention in document
- `999981284_full_001.pdf`:
    ❌ Failed to link scales to items
    ✅ Correct extraction of scales and scale names
- `999967007_full_001.pdf`: 
    ⭐ Excellently parsed despite complex structure
    ✅ Correctly transcribed nested scale hierarchy (Parent: _'Antecedents of Legitimacy'_, Child: _'Police Effectiveness'_).
- `999943323_full_001`:
    ⭐ Excellently parsed despite visually and structurally challenging format
- `999920134_full_001.pdf`:
    ❌ Failed to link scales to items
- `999915498_full_001.pdf`:
    ❌ `'item_text': '[Image of children teasing a boy]'`
- `999920818_full_001.pdf`:
    ❌ Failed to link scales to items
- `999909998_full_001`:
    ❌ Duplicated scale under different scale_id
    ❌ Failed to link scales to items
- `999906383_full_001.pdf`
    ❌ Duplicated scale under different scale_id
    ❌ Failed to link scales to items
- `999986273_full_001.pdf`:
    ⭐ Excellently parsed despite complex structure with many items

## Soft Fails / Maybes
- `999924140_full_001.pdf`: 
    ❌ Items should not be orphans, but bundled to top level scale
- `999942080_full_001.pdf`: 
    ❌ Scale-membership is merely indicated by item prefix (e.g., A1, B1, etc.). 
    ✅ Construct name not mentioned explicitley, merely in original paper title: "treatment effectiveness for patients with schizophrenia and schizoaffective".
- `999958013_full_002.pdf`:
    ✅ No items and scale measuring latent construct
    ❌ All items should be transcribed as auxiliary items
- `999943323_full_001`:
    ❌ Hallucinated response format label: `{'label': 'Definitely No', 'value': 4}]}`  should read `'Definitely Yes'` according to document, although this seems meaningless.

