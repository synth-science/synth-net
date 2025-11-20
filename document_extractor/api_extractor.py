import re
import json
import logging
import ollama
import textwrap
from enum import Enum
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, ValidationError, field_validator

from common import setup_logger, timeout_method, timeit, TimeoutError # type: ignore

logger = logging.getLogger('main')

class MaxFailedRepromptAttempts(Exception):

    def __init__(self, attempts):
        self.attempts = attempts
        super().__init__(f"Re-prompting failed consecutively after {attempts} attempts!")

class ErrorCode(Enum):
    
    INVALID_JSON = "E001"
    EMPTY_ITEMS = "E002"
    INSUFFICIENT_ITEMS = "E003"
    CONSTRUCT_NAME_TOO_LONG = "E004"
    EMPTY_ITEM_TEXT = "E005"
    VALIDATION_ERROR = "E006"
    GENERAL_ERROR = "E007"
    LOW_ITEM_TO_CONSTRUCT_RATIO = "E008"
    DUPLICATE_ITEM_TEXTS = "E009"

class ItemValidator(BaseModel):

    id: Optional[str] = None
    prefix: Optional[str] = None
    instructions: Optional[str] = None
    item_text: str
    response_format: str
    response_options: Optional[List[str]] = None
    construct_name: str = "<NONE>"
    keying: Literal["positive", "negative"] = "positive"
    keying_correction: bool = False
    
    @field_validator('construct_name')
    @classmethod
    def validate_construct_name(cls, v: str) -> str:
        if len(v) > 100:
            raise ValueError(f"construct_name too long: {len(v)} characters (max 100)")
        return v
    
    @field_validator('item_text')
    @classmethod
    def validate_item_text(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("item_text cannot be empty")
        return v
    
class SurveyValidator(BaseModel):

    thinking: str
    items: List[ItemValidator]
    
    @field_validator('items')
    @classmethod
    def validate_items(cls, v: List[ItemValidator]) -> List[ItemValidator]:
        if not v:
            raise ValueError("items list cannot be empty")
        return v
    
@dataclass
class CustomValidationError:

    code: ErrorCode
    field: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ResponseValidator:
    @staticmethod
    def validate_response(response_content: str) -> Dict[str, Any]:
        errors: List[CustomValidationError] = []
        
        try:
            json_content = json.loads(response_content)
            validated_data = SurveyValidator(**json_content)

            # CHECK: INSUFFICIENT_ITEMS
            if len(validated_data.items) <= 3:
                errors.append(CustomValidationError(
                    code=ErrorCode.INSUFFICIENT_ITEMS,
                    field='items',
                    message="Most questionnaires contain more items.",
                    details={'item_count': len(validated_data.items)}
                ))

            construct_names = set(
                item.construct_name for item in validated_data.items if item.construct_name != "<NONE>"
            )
            construct_count = len(construct_names)
            
            # CHECK: DUPLICATE_ITEM_TEXTS
            item_texts = [item.item_text for item in validated_data.items]
            duplicates = {}
            
            for idx, text in enumerate(item_texts):
                if item_texts.count(text) > 1:
                    if text not in duplicates:
                        duplicates[text] = []
                    duplicates[text].append(idx)
            
            if duplicates:
                errors.append(CustomValidationError(
                    code=ErrorCode.DUPLICATE_ITEM_TEXTS,
                    field='items',
                    message=f"Found {len(duplicates)} duplicate item text(s)",
                    details={
                        'duplicates': duplicates,
                        'duplicate_count': len(duplicates)
                    }
                ))

            # CHECK: LOW_ITEM_TO_CONSTRUCT_RATIO
            if construct_count > 0:
                item_to_construct_ratio = len(validated_data.items) / construct_count
                
                if item_to_construct_ratio < 2:
                    errors.append(CustomValidationError(
                        code=ErrorCode.LOW_ITEM_TO_CONSTRUCT_RATIO,
                        field='constructs',
                        message=f"Low item-to-construct ratio: {item_to_construct_ratio:.2f}",
                        details={
                            'item_count': len(validated_data.items),
                            'construct_count': construct_count,
                            'ratio': item_to_construct_ratio,
                            'constructs': list(construct_names)
                        }
                    ))

            if errors:
                return {
                    'is_valid': False,
                    'data': None,
                    'errors': errors
                }
            
            return {
                'is_valid': True,
                'data': validated_data.model_dump(),
                'errors': []
            }
            
        except json.JSONDecodeError as e:
            errors.append(CustomValidationError(
                code=ErrorCode.INVALID_JSON,
                field='json',
                message=f"Invalid JSON: {str(e)}",
                details={'error': str(e)}
            ))
            
        except ValidationError as e:
            for error in e.errors():
                field_path = '.'.join(str(x) for x in error['loc'])
                errors.append(CustomValidationError(
                    code=ErrorCode.VALIDATION_ERROR,
                    field=field_path,
                    message=error['msg'],
                    details={'validation_error': error}
                ))
                
        except Exception as e:
            errors.append(CustomValidationError(
                code=ErrorCode.GENERAL_ERROR,
                field='general',
                message=str(e)
            ))
            
        return {
            'is_valid': False,
            'data': None,
            'errors': errors
        }
        
    @staticmethod
    def generate_correction_prompt(errors: List[CustomValidationError]) -> str:
        
        error_descriptions = []
        
        for error in errors:
            if error.code == ErrorCode.INVALID_JSON:
                error_descriptions.append(f"- Your response contains invalid JSON: {error.message}")
                
            elif error.code == ErrorCode.INSUFFICIENT_ITEMS:
                error_descriptions.append(f"- {error.message}")
                error_descriptions.append("- Please carefully review the entire document again. Most psychological questionnaires, tests, and surveys contain multiple items.")
                error_descriptions.append("- Look for numbered questions, statements, or rating scales throughout the document.")
                error_descriptions.append("- Make sure you're not just extracting example items or section headers, but all actual test items.")

            elif error.code == ErrorCode.DUPLICATE_ITEM_TEXTS:
                duplicates = error.details.get('duplicates', {})
                error_descriptions.append(f"- {error.message}")
                error_descriptions.append("- The following item texts appear multiple times:")
                
                for text, indices in duplicates.items():
                    truncated_text = text[:100] + '...' if len(text) > 100 else text
                    error_descriptions.append(f"  * Item indices {indices}: '{truncated_text}'")
                
                error_descriptions.append("- Please ensure each item has unique text content.")
                error_descriptions.append("- If the same question appears multiple times in different contexts:")
                error_descriptions.append("  * Include the context in the item_text to make it unique")
                error_descriptions.append("  * Use the prefix field if items share common beginnings")
                error_descriptions.append("  * Check if you're mistakenly duplicating items from different sections")

            elif error.code == ErrorCode.LOW_ITEM_TO_CONSTRUCT_RATIO:
                item_count = error.details.get('item_count', 0)
                construct_count = error.details.get('construct_count', 0)
                ratio = error.details.get('ratio', 0)
                constructs = error.details.get('constructs', [])
                
                error_descriptions.append(f"- {error.message} (Found {item_count} items across {construct_count} constructs)")
                error_descriptions.append(f"- Current constructs: {', '.join(constructs)}")
                error_descriptions.append("- Please verify the number of constructs extracted. You may be:")
                error_descriptions.append("  * Over-specifying constructs (creating too many narrow categories)")
                error_descriptions.append("  * Missing items that belong to existing constructs")
                error_descriptions.append("  * Incorrectly identifying construct boundaries")
                error_descriptions.append("- Most psychological scales have multiple items per construct for reliability purposes.")
                error_descriptions.append("- Re-examine the document structure to ensure proper construct identification.")
                
            elif error.code == ErrorCode.EMPTY_ITEMS:
                error_descriptions.append("- The items list cannot be empty. Please extract all items from the questionnaire.")
                
            elif error.code == ErrorCode.CONSTRUCT_NAME_TOO_LONG:
                error_descriptions.append(f"- Field '{error.field}': {error.message}")
                error_descriptions.append("- Please use a shorter, more concise construct name.")
                
            elif error.code == ErrorCode.EMPTY_ITEM_TEXT:
                error_descriptions.append(f"- Field '{error.field}': {error.message}")
                error_descriptions.append("- Each item must have non-empty text content.")
                
            elif error.code == ErrorCode.VALIDATION_ERROR:
                error_descriptions.append(f"- Field '{error.field}': {error.message}")
                
            elif error.code == ErrorCode.GENERAL_ERROR:
                error_descriptions.append(f"- {error.message}")
                
            else:                
                error_descriptions.append(f"- {error.field}: {error.message}")
        
        prompt = f"""
        Your previous response had the following validation errors:
        
        {chr(10).join(error_descriptions)}
        
        Please correct these issues and provide a valid JSON response that:
        1. Contains all required fields
        2. Meets all field-specific constraints
        3. Is properly formatted as valid JSON
        
        Respond with the corrected JSON only.
        """
        
        return prompt.strip()

class LanguageModelExtractorAPI():

    def __init__(self, model_name: str, options: dict, max_reprompt_attempts: int, timeout_seconds: int):

        self.model_name = model_name
        self.options = options
        self.max_reprompt_attempts = max_reprompt_attempts
        self.timeout_seconds = timeout_seconds
        
        self.validator = ResponseValidator()

    @staticmethod
    def _history_to_protocol(conversation_history):
        conversation_protocol = "# Conversation Protocol\n\n"
        for index, message in enumerate(conversation_history):
            conversation_protocol += f"## Message {index} ({message['role']})\n"
            conversation_protocol += f"{message['content']}\n\n"
        
        return conversation_protocol

    @timeout_method
    def process_survey(self, transcript):

        logger.info(f"Processing survey structure via API call with `{self.model_name}`...")

        class Item(BaseModel):
            id: Optional[str] = None
            prefix: Optional[str] = None
            instructions: Optional[str] = None
            item_text: str
            response_format: str
            response_options: Optional[List[str]] = None
            construct_name: str = "<NONE>"
            keying: Literal["positive", "negative"] = "positive"
            keying_correction: bool = False

        class Survey(BaseModel):
            thinking: str
            items: List[Item]

        prompt_template = '''
            The following document is content from a survey, test or questionnaire:
            """
            {transcript}
            """

            For each item in this survey, test or questionnaire, extract the following:
            1. `id` (optional): The item identifier or number, if supplied.
            2. `instructions` (optional): Only applies if the item group or section has a different set of instructions than other item groups. 
            3. `prefix` (optional): A prefix, if multiple items are listed with only slight variations but share the same prepended substring.
                - e. g.,  Prefix: "During the last month, I've felt...." Item Text: "...happy and excited about the future".
            4. `item_text`: The item text, statement or question presented to the survey respondent.
            5. `response_format`: The response format used to record the survey participants response.
                - e.g., "rating-scale", "likert-scale", "multiple-choice", "single-choice", "open-ended", "numeric input", "ranking", "unknown", etc.
            6. `response_options` (optional): List of **categorical** response options, **if** presented with the item.
                - Must be **categorical** ("headaches", "anger", "pain", etc.) and **not ordinal** (e.g., "strongly agree", "agree", "disagree", etc.)
                - Only extract if the `response_options` differ for the remaining items.
            7. `construct_name`: The **construct name** measured by the item (e.g., construct, scale, dimension).
                - Make sure that the document states explicitly that the item belongs to the scale, which may appear as:
                    * Section headings or subheadings
                    * Bold or italic
                    * Explicitly labeled dimensions
                    * Grouped item categories
                    * In table columns, rows or headers
                - If no construct names or scales are explicitly mentioned, infer a single construct name from the document title.
                - If multiple hierarchical levels are mentioned (e.g., "Big Five Personality > Extraversion"), use the most specific level.
                - Infer the name of the **construct**, not the **scale name** (e.g., "Rumination" instead of "rumination scale").
                - **Special cases**:
                    * For demographic items (age, gender, education, etc.): use "Demographics"
                    * For items that cannot be associated with any construct: use "<NONE>"
            8. `keying`: The **keying** of the *item* in relation to the *construct* (i. e., positively-keyed or negatively-keyed/reverse-keyed).
                - Valid values are `positive` or `negative`.
                - If keying is not explicitly stated, look for:
                    * Scoring instructions (e.g., "items 2, 5, 7 are reverse-scored")
                    * Notation systems (e.g., asterisks, "(R)" markers, or minus signs indicating reverse items)
                    * Coding schemes showing how responses should be converted
                    * Score calculation formulas
                - If no keying information is explicitly mentioned in the document, assume `positive`.
                - Only correct the keying direction in **rare cases** and when you are **highly confident** that the document's explicit information is incorrect.
            9. `keying_correction`: `true` if `keying` needed correciton, else `false`.
            Note: If an item belongs to more than one scale, add it as an new object.            

            **Ensure that all non-optional properties are extracted without exception!**
            Respond with valid JSON array only, and nothing else.
        '''
        result = {
            'survey_reasoning_attempts': 1,
            'survey_reasoning': "",
            'survey_duration_seconds': 0,
            'survey_validation_errors': [],
            'survey_content': None,
            'survey_length': None,
            'survey_has_error': False,
            'survey_error': None,            
            'conversation_protocol': None
        }

        initial_prompt = textwrap.dedent(prompt_template.format(transcript=transcript))
        conversation_history = [{'role': 'user', 'content': initial_prompt}]
        
        max_attempts = 2
        has_reprompted = False

        for attempt in range(max_attempts):
            current_attempt = attempt + 1
            result['survey_reasoning_attempts'] = current_attempt
            
            try:
                response = ollama.chat(
                    model=self.model_name,
                    options=self.options,
                    messages=conversation_history,
                    format=Survey.model_json_schema()
                )

                result['survey_duration_seconds'] += response.total_duration * 1e-9
                response_message = response.message['content']

                try:
                    response_json = json.loads(response_message)
                    thinking = response_json.get('thinking', '[NO THOUGHTS]')
                except json.JSONDecodeError:
                    thinking = 'Failed to parse JSON for reasoning'
                
                attempt_prefix = f"\n{'#'*5} Attempt {current_attempt} {'#'*5}\n"
                reasoning_attempt = attempt_prefix + str(thinking)
                result['survey_reasoning'] += reasoning_attempt

                response_validation = self.validator.validate_response(response_message)

                result['survey_validation_errors'].extend(
                    [f"{error.field}: {error.message}" for error in response_validation['errors']]
                )

                if response_validation['is_valid']:

                    data = response_validation['data']
                    result['survey_content'] = data['items']
                    result['survey_length'] = len(data['items'])                                        
                    result['conversation_protocol'] = self._history_to_protocol(conversation_history)

                    logger.info(f"Successfully extracted survey structure after {current_attempt} attempt(s)!")
                    return result

                elif not has_reprompted and current_attempt < max_attempts:

                    has_reprompted = True
                    correction_prompt = self.validator.generate_correction_prompt(
                        response_validation['errors']
                    )

                    conversation_history.append({
                        'role': 'assistant',
                        'content': response_message
                    })

                    conversation_history.append({
                        'role': 'user',
                        'content': correction_prompt
                    })

                    logger.warning(
                        f"Survey validation failed on attempt {current_attempt}. "
                        f"Errors: {[error.message for error in response_validation['errors']]}. "
                        f"Re-prompting once..."
                    )
                    continue

                else:

                    logger.warning(
                        f"Survey validation failed after re-prompt on attempt {current_attempt}. "
                        f"Accepting response with validation errors: {[error.message for error in response_validation['errors']]}"
                    )
                    
                    try:
                        response_json = json.loads(response_message)
                        if 'items' in response_json and response_json['items']:
                            result['survey_content'] = response_json['items']
                            result['survey_length'] = len(response_json['items'])
                        else:
                            result['survey_content'] = []
                            result['survey_length'] = 0
                    except (json.JSONDecodeError, KeyError, TypeError):
                        result['survey_content'] = []
                        result['survey_length'] = 0
                    
                    result['conversation_protocol'] = self._history_to_protocol(conversation_history)
                    
                    logger.info(f"Completed survey extraction with validation errors after {current_attempt} attempt(s)")
                    return result
                
            except Exception as e:
                logger.error(f"Error on prompt attempt {current_attempt}: {str(e)}")
                result['survey_has_error'] = True
                result['survey_error'] = str(e)
                result['conversation_protocol'] = self._history_to_protocol(conversation_history)
                return result

        result['conversation_protocol'] = self._history_to_protocol(conversation_history)
        return result