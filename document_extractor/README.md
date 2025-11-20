# Document Extractor

A tool for extracting structured survey data from PDF documents in the APA PsycTests database using vision-language models and LLM processing.

## Overview

The Document Extractor is a two-stage pipeline that processes psychological test documents:

1. **Vision-Language Model Extraction**: Uses Qwen2.5-VL-7B-Instruct to transcribe PDF pages into markdown format, preserving document structure, tables, and formatting.

2. **LLM Processing**: Processes the transcribed content using an Ollama-powered LLM (Qwen3:32b) to extract structured survey data, including:
   - Individual test items
   - Response formats and options
   - Construct names/scales
   - Item keying information
   - Instructions and prefixes

The pipeline includes robust error handling, timeout management, and automatic re-prompting for validation failures.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the required model specified in the config via Ollama (e.g., qwen3:32b):
```bash
ollama pull qwen3:32b
```

3. Download the Qwen2.5-VL-7B-Instruct model to the specified path in the config.

## Usage

Run the extraction pipeline:

```bash
python main.py
```

The pipeline will:
1. Process all PDF files in the input directory
2. Save results incrementally to both JSON and Parquet formats
3. Skip already processed files if output exists (unless overwrite is enabled)

## Output

The pipeline produces records with the following properties:

### Record Structure

| Property | Type | Description |
|----------|------|-------------|
| `filename` | string | Name of the source PDF file |
| `cover_text` | string | Extracted text from the cover page |
| `text` | string | Raw text extracted from the document (if available) |
| `duration` | float | Total processing time in seconds |
| **Transcript Properties** | | |
| `transcript_content` | string | Markdown-formatted transcription of the document |
| `transcript_duration_seconds` | float | Time taken for transcription |
| `transcript_length` | integer | Character count of the transcription |
| `transcript_has_error` | boolean | Whether an error occurred during transcription |
| `transcript_error` | string \| null | Error message if transcription failed |
| **Survey Properties** | | |
| `survey_reasoning_attempts` | integer | Number of attempts to extract valid survey data |
| `survey_reasoning` | string | LLM's reasoning process during extraction |
| `survey_duration_seconds` | float | Time taken for survey extraction |
| `survey_validation_errors` | array | List of validation errors encountered |
| `survey_content` | array \| null | Extracted survey items (see structure below) |
| `survey_length` | integer \| null | Number of items extracted |
| `survey_has_error` | boolean | Whether an error occurred during extraction |
| `survey_error` | string \| null | Error message if extraction failed |
| `conversation_protocol` | string \| null | Full conversation history with LLM |

### Survey Content Item Structure

Each item in `survey_content` contains:

| Property | Type | Description |
|----------|------|-------------|
| `id` | string \| null | Item identifier or number |
| `prefix` | string \| null | Common prefix shared by multiple items |
| `instructions` | string \| null | Special instructions for item group |
| `item_text` | string | The actual item text/question |
| `response_format` | string | Type of response format (e.g., "likert-scale", "multiple-choice") |
| `response_options` | array \| null | List of categorical response options |
| `construct_name` | string | Name of construct/scale measured (or "<NONE>") |
| `keying` | string | "positive" or "negative" indicating item scoring direction |
| `keying_correction` | boolean | Whether keying was corrected from document |

## Validation Rules

The extractor enforces several validation rules:

1. **Minimum Items**: Surveys must contain more than 3 items
2. **No Duplicate Items**: Each item text must be unique
3. **Sufficient Item-to-Construct Ratio**: At least 2 items per construct
4. **Valid JSON**: Output must be properly formatted JSON
5. **Required Fields**: All non-optional fields must be present
6. **Field Constraints**: Construct names limited to 100 characters

When validation fails, the system automatically generates error-specific correction prompts and retries up to the configured maximum attempts.

## Error Handling

- **Timeouts**: Each processing step has configurable timeout protection
- **Validation Failures**: Automatic re-prompting with specific error feedback
- **Processing Errors**: Graceful handling with error logging and partial result saving
- **Resume Capability**: Failed files can be reprocessed in subsequent runs