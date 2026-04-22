import re
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr, computed_field, model_validator


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class ResponseOption(BaseModel):
    """One selectable answer within a ResponseFormat."""
    value: Union[int, float, str] = Field(
        description="Numeric code for scored options (e.g. 1–5), string for categorical codes."
    )
    label: str = Field(
        description="Verbal anchor shown to the respondent, e.g. 'Strongly Disagree'."
    )


class ResponseFormat(BaseModel):
    """Answer format shared by one or more items, defined once and referenced by id."""
    format_id: int = Field(description="Unique integer id within the survey.")
    format_type: Literal[
        "rating_scale", "single_choice", "multiple_choice", "binary", "open_ended", "other",
    ] = Field(description="Kind of answer expected from the respondent.")
    options: Optional[List[ResponseOption]] = Field(
        default=None,
        description="Selectable options; None for open-ended formats.",
    )
    min_value: Optional[float] = Field(
        default=None,
        description="Lowest numeric option value; enables reverse-keying. None if non-numeric.",
    )
    max_value: Optional[float] = Field(
        default=None,
        description="Highest numeric option value; enables reverse-keying. None if non-numeric.",
    )


class ScoredItem(BaseModel):
    """A psychometric item, listed inline inside the Scale that scores it.

    The same item may appear inside more than one scale (e.g. a facet item
    that also rolls up into a parent composite). Duplicate ScoredItems with
    identical item_text are collapsed into a single logical item by Survey
    validation, which also stamps a stable `item_id` shared by all duplicates.
    """
    item_text: str = Field(
        description="Verbatim wording of the question."
    )
    item_comments: Optional[str] = Field(
        default=None,
        description="Information not intended for display to the respondent e.g., instructor notes.",
    )
    response_format_id: int = Field(
        description="format_id of the ResponseFormat this item uses."
    )
    language: str = Field(
        description="ISO 639-1 code of the original item text (e.g. 'en', 'de', 'es')."
    )
    is_translated: bool = Field(
        default=False,
        description="True if item_text is an English translation rather than the original wording.",
    )
    has_image: bool = Field(
        default=False,
        description="True if the item is presented with an image.",
    )
    image_description: Optional[str] = Field(
        default=None,
        description="Description of the image, if `has_image` is True.",
    )
    reverse_keyed: bool = Field(
        default=False,
        description="True if the response is reversed before scoring this scale.",
    )

    # Synthetic id assigned post-extraction; hidden from the LLM-facing schema.
    _item_id: int = PrivateAttr(default=0)

    @computed_field
    @property
    def item_id(self) -> int:
        return self._item_id


class AuxiliaryItem(BaseModel):
    """A non-psychometric question (demographics, admin, clinical context) excluded from scoring."""
    item_id: int = Field(
        description="Unique integer id within auxiliary_items."
    )
    item_text: str = Field(
        description="Verbatim wording of the question."
    )
    response_format_id: Optional[int] = Field(
        default=None,
        description="format_id of the ResponseFormat, or None for free-text/date fields without fixed options.",
    )
    category: Optional[str] = Field(
        default=None,
        description="Free-form tag grouping the item, e.g. 'demographic', 'admin', 'clinical_context'.",
    )
    language: str = Field(
        description="ISO 639-1 code of the original item text."
    )
    is_translated: bool = Field(
        default=False,
        description="True if item_text is an English translation rather than the original.",
    )


class Scale(BaseModel):
    """A scale, subscale, or composite construct.

    Scales form a tree: top-level scales appear in Survey.scales, and any
    finer-grained facets are nested under their parent via `subscales`.
    Every scale lists the items it scores inline in `items`. A purely
    composite scale (e.g. a domain whose total = sum of its facets) re-lists
    those items here; identical item_text across scales is collapsed to a
    single logical item by Survey validation (same synthetic item_id),
    so the duplication is logical, not data.
    """
    scale_id: int = Field(
        description="Unique integer id within the survey."
    )
    scale_name: str = Field(
        description="Name as printed in the instrument, e.g. 'Extraversion'."
    )
    construct_name: str = Field(
        description="Underlying psychological construct the scale measures."
    )
    report_type: Literal["self-report", "other-report"] = Field(
        default="self-report",
        description="Whether the scale is intended for respondents to report on themselves or on someone else (e.g. a child, patient, or friend).",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Scale-specific instructions if printed separately from survey-level ones.",
    )
    items_prefix: Optional[str] = Field(
        default=None,
        description="Stem prepended to each item, e.g. 'I see myself as someone who...'.",
    )
    items_context: Optional[str] = Field(
        default=None,
        description="Contextual information about the items contributing to this scale, e.g. 'In the past two weeks, how often have you felt...'.",
    )
    items: List[ScoredItem] = Field(
        min_length=1,
        description=(
            "Items scored into this scale, listed inline. MUST contain at "
            "least one item. If the same item is also scored into another "
            "scale, list it inside that scale too with identical item_text — "
            "duplicates are collapsed to a single logical item by validation."
        ),
    )
    subscales: List["Scale"] = Field(
        default_factory=list,
        description="Facets or subscales nested under this scale.",
    )
    language: str = Field(
        description="ISO 639-1 code of the original scale name/instructions."
    )
    is_translated: bool = Field(
        default=False,
        description="True if scale texts are English translations rather than the original.",
    )


class Survey(BaseModel):
    """A complete behavioral measurement instrument."""
    thinking: str = Field(
        description="LLM scratch field for extraction reasoning; ignored downstream."
    )
    survey_name: str = Field(
        description="Full name of the instrument, e.g. 'Big Five Inventory 2'."
    )
    survey_abbreviation: Optional[str] = Field(
        default=None,
        description="Abbreviated name of the instrument, e.g. 'BFI-2'."
    )
    survey_version: Optional[str] = Field(
        default=None,
        description="Version of the instrument if specified, e.g., 'Short Form'."
    )
    instructions: Optional[str] = Field(
        default=None,
        description="General instructions shown to respondents at the start of the instrument.",
    )
    items_prefix: Optional[str] = Field(
        default=None,
        description="Survey-wide stem prepended to every item unless a scale overrides it.",
    )
    response_formats: List[ResponseFormat] = Field(
        description="All response formats used anywhere in the survey."
    )
    auxiliary_items: List[AuxiliaryItem] = Field(
        default_factory=list,
        description="Non-scored questions (demographics, admin, clinical context) kept separate from scale items.",
    )
    scales: List[Scale] = Field(
        description=(
            "Top-level scales only; facets/subscales are nested inside "
            "their parent via `Scale.subscales`. Items live inside each "
            "scale via `Scale.items` — there is no separate top-level "
            "items list. Items only exist in the context of a scale."
        ),
    )
    language: str = Field(
        description="ISO 639-1 code of the original instrument (e.g. 'en', 'de', 'es')."
    )
    is_translated: bool = Field(
        default=False,
        description="True if any text in the survey has been transcribed as an English translation.",
    )

    @model_validator(mode="after")
    def _validate(self):
        def walk(scales: List[Scale]):
            for s in scales:
                yield s
                yield from walk(s.subscales)

        all_scales = list(walk(self.scales))

        scale_ids = [s.scale_id for s in all_scales]
        if len(scale_ids) != len(set(scale_ids)):
            raise ValueError("scale_ids must be unique across the scale tree")

        fmt_ids = [f.format_id for f in self.response_formats]
        if len(fmt_ids) != len(set(fmt_ids)):
            raise ValueError("response_format ids must be unique")
        valid_fmt_ids = {f.format_id for f in self.response_formats}

        # Assign synthetic item_ids by deduping on normalized item_text. The
        # LLM never sees these ids — they're materialised here so downstream
        # code (tests/utils, parquet consumers) can refer to a logical item.
        text_to_id: dict[str, int] = {}
        groups: dict[str, list[ScoredItem]] = {}
        next_id = 1
        for s in all_scales:
            for si in s.items:
                if si.response_format_id not in valid_fmt_ids:
                    raise ValueError(
                        f"item in scale {s.scale_id} ({s.scale_name!r}) "
                        f"references unknown response_format_id {si.response_format_id}"
                    )
                key = _normalize(si.item_text)
                if key not in text_to_id:
                    text_to_id[key] = next_id
                    next_id += 1
                si._item_id = text_to_id[key]
                groups.setdefault(key, []).append(si)

        # Cross-scale consistency: a logical item must look identical
        # everywhere it's listed. Diverging attributes almost always mean
        # the LLM transcribed the same item slightly differently in two
        # scales; surface that loudly rather than silently corrupting data.
        shared_fields = (
            "response_format_id",
            "language",
            "is_translated",
            "has_image",
            "image_description",
        )
        for key, group in groups.items():
            if len(group) < 2:
                continue
            head = group[0]
            for other in group[1:]:
                for field_name in shared_fields:
                    if getattr(head, field_name) != getattr(other, field_name):
                        raise ValueError(
                            f"item {head.item_text!r} appears in multiple scales but "
                            f"{field_name!r} differs across occurrences "
                            f"({getattr(head, field_name)!r} vs {getattr(other, field_name)!r})"
                        )

        return self


Scale.model_rebuild()
