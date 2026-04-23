import os
import ast
import random
import yaml
import json
import fitz
import ollama
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from models import Survey
from pdb import set_trace as trace


def load_config(config_path: str) -> dict:
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    return config


def setup_logging():
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file_path = logs_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )


def load_datamodel_docs(documentation_path=None) -> str:
    """Distils the survey data-model documentation to a simplified form for prompting."""

    def _is_validator(stmt) -> bool:
        if not isinstance(stmt, ast.FunctionDef):
            return False
        for d in stmt.decorator_list:
            func = d.func if isinstance(d, ast.Call) else d
            if getattr(func, "id", None) == "model_validator":
                return True
        return False


    def _is_thinking_field(stmt) -> bool:
        return (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "thinking"
        )
    tree = ast.parse(Path(documentation_path).read_text())

    tree.body = [
        n for n in tree.body
        if not (isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "attr", None) == "model_rebuild")
    ]

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            node.body = [
                s for s in node.body
                if not _is_validator(s) and not _is_thinking_field(s)
            ] or [ast.Pass()]

    return ast.unparse(tree)


def convert_pdf_to_image(pdf_path):

    pdf_document = fitz.open(pdf_path)
    pdf_images = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        zoom_matrix = fitz.Matrix(2.0, 2.0)
        pixmap = page.get_pixmap(matrix=zoom_matrix)

        img_bytes = pixmap.tobytes("png")
        pdf_images.append(img_bytes)

    return pdf_images


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


@dataclass
class ExtractionResult:
    """Result of a single extract_survey() call.

    `survey` is None if the LLM response failed schema validation.
    Metadata is always populated so callers can persist timing / raw output
    even when validation fails.
    """
    filename: str
    survey: Optional[Survey]
    # survey_json: Optional[dict]
    raw_response: str
    response_metrics: dict = field(default_factory=dict)
    validation_error: Optional[str] = None
    timestamp: str = ""


def extract_survey(pdf_path: Path | str, config: dict) -> ExtractionResult:
    """Run one extraction pass on a PDF. Pure function — no logging setup,
    no filesystem writes. Callable from both main() and the test harness.
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name

    pdf_images = convert_pdf_to_image(str(pdf_path))

    template_dir = Path(__file__).parent / "templates"
    datamodel_path = Path(__file__).parent / "models.py"
    datamodel_docs = load_datamodel_docs(datamodel_path)

    system_prompt_dict = {
        "role": "system",
        "content": read_file(template_dir / "system.md"),
    }
    user_prompt = "{prompt}\n\n{documentation}".format(
        prompt=read_file(template_dir / "user.md"),
        documentation=datamodel_docs,
    )
    user_prompt_dict = {
        "role": "user",
        "content": user_prompt,
        "images": pdf_images,
    }

    chat_kwargs: dict[str, Any] = {
        "model": config.get("model", None),
        "messages": [system_prompt_dict, user_prompt_dict],
        "format": Survey.model_json_schema(),
    }
    if config.get("options"):
        chat_kwargs["options"] = config["options"]

    response = ollama.chat(**chat_kwargs)

    response_message = response.message["content"]

    survey_obj: Optional[Survey] = None
    # survey_json: Optional[dict] = None
    validation_error: Optional[str] = None
    try:
        survey_obj = Survey.model_validate_json(response_message)
        # Dump the validated model so the parquet carries synthetic item_ids
        # (Survey validation stamps the same id onto duplicate ScoredItems).
        # survey_json = survey_obj.model_dump(exclude={"thinking"})
    except json.JSONDecodeError as e:
        validation_error = f"JSON decoding error: {e}"
    except Exception as e:
        validation_error = f"Schema validation error: {e}"

    input_token_count = response.get('prompt_eval_count', 0)
    output_token_count = response.get('eval_count', 0)
    total_token_count = input_token_count + output_token_count
    return ExtractionResult(
        filename=filename,
        survey=survey_obj,
        # survey_json=survey_json,
        raw_response=response_message,
        response_metrics={
            "created_at": response.created_at,
            "total_duration": response.total_duration,
            "load_duration": response.load_duration,
            "prompt_eval_duration": response.prompt_eval_duration,
            "eval_duration": response.eval_duration,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "total_token_count": total_token_count,
        },
        validation_error=validation_error,
        timestamp=datetime.now().isoformat(),
    )


def main():
    setup_logging()
    config = load_config(config_path="./config.yaml")

    input_files = os.listdir(path=config['input_dir'])
    random.seed(config['seed'])
    random.shuffle(input_files)
    filenames = [x for x in input_files if x.endswith('.pdf')]

    records = []

    filenames = ["999941280_full_001.pdf"] # debug
    for filename in tqdm(filenames):
        logging.info(f"Processing file: {filename}")
        pdf_path = os.path.join(config['input_dir'], filename)

        logging.info(f"Sending document to model for extraction...")
        trace() # debug
        try:
            result = extract_survey(pdf_path, config)
        except Exception as e:
            logging.error(f"Error during model inference for file {filename}: {e}")
            continue

        logging.info(f"Received response from model for file {filename}")
        if result.validation_error:
            logging.error(f"{filename}: {result.validation_error}")
        trace() # debug
        records.append({
            "filename": result.filename,
            # "survey_json": json.dumps(result.survey_json),
            "response": result.raw_response,
            "response_created_at": result.response_metrics["created_at"],
            "response_total_duration": result.response_metrics["total_duration"],
            "response_load_duration": result.response_metrics["load_duration"],
            "response_prompt_eval_duration": result.response_metrics["prompt_eval_duration"],
            "response_eval_duration": result.response_metrics["eval_duration"],
            "timestamp": result.timestamp,
        })

        try:
            pd.DataFrame(records).to_parquet(config["output_path"])
        except Exception as e:
            logging.error(f"Failed saving records as .parquet: {e}")


if __name__ == "__main__":
    main()
