import os
import ast
import random
import yaml
import json
import fitz
import ollama
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from models import Survey
from pdb import set_trace as trace

def load_config(config_path: str) -> dict:
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    return config

def setup_logging():
    log_file_path = Path(__file__).parent / "logfile.log"
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


setup_logging()
config = load_config(config_path="./config.yaml")

input_files = os.listdir(path=config['input_dir'])
random.seed(config['seed'])
random.shuffle(input_files)
filenames = [x for x in input_files if x.endswith('.pdf')]

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

records = []

for filename in tqdm(filenames):

    logging.info(f"Processing file: {filename}")
    pdf_path = os.path.join(config['input_dir'], filename)
    pdf_images = convert_pdf_to_image(pdf_path)

    template_dir = Path(__file__).parent / "../templates/"

    system_prompt_dict = {
        "role": "system",
        "content": f'{read_file(template_dir / "system.md")}'
    }

    datamodel_path = Path(__file__).parent / "../models.py"
    datamodel_docs = load_datamodel_docs(datamodel_path)

    user_prompt = "{prompt}\n\n{documentation}".format(
        prompt=f'{read_file(template_dir / "user.md")}',
        documentation=datamodel_docs
    )

    user_prompt_dict = {
        "role": "user",
        "content": user_prompt,
        "images": pdf_images
    }

    logging.info(f"Sending document to model for extraction...")
    trace()
    try:
        response = ollama.chat(
            model='gemma4:31b',
            messages=[
                system_prompt_dict, 
                user_prompt_dict
            ],
            format=Survey.model_json_schema()
        )
    except Exception as e:
        logging.error(f"Error during model inference for file {filename}: {e}")
        continue

    logging.info(f"Received response from model for file {filename}")
    response_message = response.message['content']
    thinking = response.message.get('thinking', '')
    survey_json = None

    try:
        if Survey.model_validate_json(response_message):
            survey_json = json.loads(response_message)
            survey_json.pop("thinking", None)
        else:
            raise ValueError("Response does not conform to Survey schema")
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error for file {filename}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error for file {filename}: {e}")

    records.append({
        "filename": filename,
        "survey_json": json.dumps(survey_json),
        "response": response_message,
        "response_created_at": response.created_at,
        "response_total_duration": response.total_duration,
        "response_load_duration": response.load_duration,
        "response_prompt_eval_duration": response.prompt_eval_duration,
        "response_eval_duration": response.eval_duration,
        "timestamp": datetime.now().isoformat(),        
    })

    try:
        pd.DataFrame(records).to_parquet(config["output_path"])
    except Exception as e:
        logging.error(f"Failed saving records as .parquet: {e}")