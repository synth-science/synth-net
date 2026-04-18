import os
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

log_file = Path(__file__).parent / "logfile.log"

config_path = "./config.yaml"
with open(config_path) as stream:
    config = yaml.safe_load(stream)

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

def read_file(filename):
    script_dir = Path(__file__).parent
    file_path = script_dir / filename

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content

records = []

for filename in tqdm(filenames):

    logging.info(f"Processing file: {filename}")
    pdf_path = os.path.join(config['input_dir'], filename)
    pdf_images = convert_pdf_to_image(pdf_path)

    user_prompt_template = read_file("user_prompt.md")
    model_documentation = read_file("../models.py")

    system_prompt = {
        "role": "system",
        "content": read_file("system_prompt.md")
    }
    user_prompt = {
        "role": "user",
        "content": f"{user_prompt_template}\n\n{model_documentation}",
        "images": pdf_images
    }

    logging.info(f"Sending document to model for extraction...")
    try:
        response = ollama.chat(
            model='gemma4:31b',
            messages=[system_prompt, user_prompt],
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

    records = {
        "filename": filename,
        "survey_json": survey_json,
        "response": response_message,
        "response_created_at": response.created_at,
        "response_total_duration": response.total_duration,
        "response_load_duration": response.load_duration,
        "response_prompt_eval_duration": response.prompt_eval_duration,
        "response_eval_duration": response.eval_duration,
        "timestamp": datetime.now().isoformat(),        
    }

    try:
        pd.DataFrame(records).to_parquet(config["output_path"])
    except Exception as e:
        logging.error(f"Failed saving records as .parquet: {e}")