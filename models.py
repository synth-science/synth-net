from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


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


class Item(BaseModel):
    """A psychometric item that is scored into one or more scales."""
    item_id: int = Field(
        description="Unique id, not shared with any AuxiliaryItem."
    )
    item_text: str = Field(
        description="Verbatim wording of the question."
    )
    item_comments: Optional[str] = Field(
        default=None,
        description="Information not intended for display to the respondent e.g., instructor notes."
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


class AuxiliaryItem(BaseModel):
    """A non-psychometric question (demographics, admin, clinical context) excluded from scoring."""
    item_id: int = Field(
        description="Unique id, not shared with any Item."
    )
    item_text: str = Field(
        description="Verbatim wording of the question.")
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

class ScoredItem(BaseModel):
    """An item's contribution to a single scale."""
    item_id: int = Field(
        description="item_id of the Item contributing to this scale."
    )
    reverse_keyed: bool = Field(
        default=False,
        description="True if the response is reversed before scoring this scale.",
    )

class Scale(BaseModel):
    """A scale, subscale, or composite construct.

    Scales form a tree: top-level scales appear in Survey.scales, and any
    finer-grained facets are nested under their parent via `subscales`.
    A scale may have `scored_items`, `subscales`, or both. A purely
    composite scale (e.g. a domain scored as the mean of its facets) has
    `subscales` but no `scored_items`.
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
    scored_items: List[ScoredItem] = Field(
        default_factory=list,
        description=(
            "Items directly scored into this scale. Leave empty only if "
            "this scale is purely composed of subscales."
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
    items: List[Item] = Field(
        description=(
            "All psychometric items, listed once. Scales reference these "
            "by item_id; every item here MUST be referenced by at least "
            "one scale (or subscale) in `scales`."
        ),
    )
    auxiliary_items: List[AuxiliaryItem] = Field(
        default_factory=list,
        description="Non-scored questions (demographics, admin, clinical context) kept separate from `items`.",
    )
    scales: List[Scale] = Field(
        description=(
            "Top-level scales only. Facets/subscales are nested inside "
            "their parent via `Scale.subscales`."
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
        # Collect scales recursively
        def walk(scales: List[Scale]):
            for s in scales:
                yield s
                yield from walk(s.subscales)

        all_scales = list(walk(self.scales))

        # Unique scale_ids across the whole tree
        scale_ids = [s.scale_id for s in all_scales]
        if len(scale_ids) != len(set(scale_ids)):
            raise ValueError("scale_ids must be unique across the scale tree")

        # Unique format ids
        fmt_ids = [f.format_id for f in self.response_formats]
        if len(fmt_ids) != len(set(fmt_ids)):
            raise ValueError("response_format ids must be unique")

        # Unique item ids across items + auxiliary_items
        item_ids = [i.item_id for i in self.items] + [a.item_id for a in self.auxiliary_items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("item_ids must be unique across items and auxiliary_items")

        # Every Item must be referenced by at least one scale
        valid_item_ids = {i.item_id for i in self.items}
        referenced = {si.item_id for s in all_scales for si in s.scored_items}

        unknown = referenced - valid_item_ids
        if unknown:
            raise ValueError(f"Scales reference unknown item_ids: {sorted(unknown)}")

        orphans = valid_item_ids - referenced
        if orphans:
            raise ValueError(
                f"Psychometric items must belong to at least one scale; "
                f"orphan item_ids: {sorted(orphans)}"
            )

        # Item response_format_ids must exist
        valid_fmt_ids = {f.format_id for f in self.response_formats}
        for i in self.items:
            if i.response_format_id not in valid_fmt_ids:
                raise ValueError(f"Item {i.item_id} references unknown response_format_id")

        # A scale must either score items directly or have subscales
        for s in all_scales:
            if not s.scored_items and not s.subscales:
                raise ValueError(
                    f"Scale {s.scale_id} ({s.scale_name}) has neither "
                    f"scored_items nor subscales; nothing to score."
                )

        return self


Scale.model_rebuild()