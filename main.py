import os
import ast
import random
import time
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


def setup_logging() -> Path:
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file_path = logs_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ],
        force=True,
    )
    return log_file_path


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
    even when validation fails. `retry_count` is the number of extra
    ollama.chat rounds beyond the first (0 on a clean first-shot).
    """
    filename: str
    survey: Optional[Survey]
    raw_response: str
    response_metrics: dict = field(default_factory=dict)
    validation_error: Optional[str] = None
    retry_count: int = 0
    success: bool = False
    timestamp: str = ""


def extract_survey(pdf_path: Path | str, config: dict) -> ExtractionResult:
    """Run one extraction pass on a PDF. Pure function — no logging setup,
    no filesystem writes. Callable from both main() and the test harness.
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name

    pdf_images = convert_pdf_to_image(str(pdf_path))
    logging.info(f"converted {filename}: {len(pdf_images)} pages")

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

    error_preamble = read_file(template_dir / "error.md")
    max_retries = int(config.get("max_retries", 0))

    messages: list[dict[str, Any]] = [system_prompt_dict, user_prompt_dict]
    metrics: dict[str, Any] = {
        "created_at": None,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_duration": 0,
        "eval_duration": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "total_token_count": 0,
    }
    survey_obj: Optional[Survey] = None
    validation_error: Optional[str] = None
    raw_response: str = ""
    attempt = 0

    attempt_timeout = float(config.get("attempt_timeout_secs", 1200))
    rate_log_secs = float(config.get("rate_log_secs", 10))
    logging.info(
        f"starting extraction for {filename} (max_retries={max_retries}, "
        f"timeout={attempt_timeout:.0f}s)"
    )
    for attempt in range(max_retries + 1):
        chat_kwargs: dict[str, Any] = {
            "model": config.get("model", None),
            "messages": messages,
            "format": Survey.model_json_schema(),
            "stream": True,
        }
        if config.get("options"):
            chat_kwargs["options"] = config["options"]
        if config.get("keep_alive") is not None:
            chat_kwargs["keep_alive"] = config["keep_alive"]

        t0 = time.monotonic()
        # Stream chunks so we can (a) log generation rate during long calls,
        # (b) bail out cleanly if the call runs past the per-attempt cap, and
        # (c) dump the partial output for offline inspection on timeout.
        chunks: list[str] = []
        chars = 0
        chunk_count = 0
        last_log = t0
        final_chunk: Any = None
        timed_out = False
        stream = ollama.chat(**chat_kwargs)
        try:
            for chunk in stream:
                chunk_count += 1
                msg = getattr(chunk, "message", None)
                content = getattr(msg, "content", "") or "" if msg is not None else ""
                if content:
                    chunks.append(content)
                    chars += len(content)
                if chunk.done:
                    final_chunk = chunk
                now = time.monotonic()
                if now - t0 > attempt_timeout:
                    timed_out = True
                    break
                if now - last_log >= rate_log_secs:
                    elapsed = now - t0
                    rate = chars / elapsed if elapsed > 0 else 0.0
                    tail = "".join(chunks)[-50:].replace("\n", "\\n")
                    logging.info(
                        f"stream attempt={attempt} t={elapsed:.0f}s "
                        f"chunks={chunk_count} chars={chars} rate={rate:.1f}c/s "
                        f"tail={tail!r}"
                    )
                    last_log = now
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        wall = time.monotonic() - t0
        raw_response = "".join(chunks)

        if timed_out:
            diag_dir = Path(__file__).parent / "logs" / "diag"
            diag_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            stem = Path(filename).stem
            partial_path = diag_dir / f"{ts}_{stem}_attempt{attempt}.partial.json"
            partial_path.write_text(raw_response, encoding="utf-8")
            logging.error(
                f"attempt={attempt} TIMED OUT after {wall:.1f}s "
                f"(cap={attempt_timeout:.0f}s) "
                f"chunks={chunk_count} chars={chars} "
                f"partial saved at {partial_path}"
            )
            raise TimeoutError(
                f"extraction timeout after {wall:.0f}s "
                f"(cap {attempt_timeout:.0f}s); partial at {partial_path}"
            )

        in_tok = (getattr(final_chunk, "prompt_eval_count", 0) or 0) if final_chunk else 0
        out_tok = (getattr(final_chunk, "eval_count", 0) or 0) if final_chunk else 0
        load_dur = (getattr(final_chunk, "load_duration", 0) or 0) if final_chunk else 0
        prefill_dur = (getattr(final_chunk, "prompt_eval_duration", 0) or 0) if final_chunk else 0
        gen_dur = (getattr(final_chunk, "eval_duration", 0) or 0) if final_chunk else 0
        total_dur = (getattr(final_chunk, "total_duration", 0) or 0) if final_chunk else 0
        logging.info(
            f"chat attempt={attempt} wall={wall:.1f}s "
            f"prompt_tok={in_tok} "
            f"eval_tok={out_tok} "
            f"chunks={chunk_count} "
            f"load_ms={load_dur / 1e6:.0f} "
            f"prefill_ms={prefill_dur / 1e6:.0f} "
            f"gen_ms={gen_dur / 1e6:.0f}"
        )

        if final_chunk is not None:
            metrics["created_at"] = getattr(final_chunk, "created_at", None)
        metrics["total_duration"] += total_dur
        metrics["load_duration"] += load_dur
        metrics["prompt_eval_duration"] += prefill_dur
        metrics["eval_duration"] += gen_dur
        metrics["input_token_count"] += in_tok
        metrics["output_token_count"] += out_tok
        metrics["total_token_count"] += in_tok + out_tok

        try:
            survey_obj = Survey.model_validate_json(raw_response)
            validation_error = None
            logging.info(f"attempt={attempt} validated successfully")
            break
        except json.JSONDecodeError as e:
            validation_error = f"JSON decoding error: {e}"
            logging.warning(f"attempt={attempt} JSON decode failed: {e}")
        except Exception as e:
            validation_error = f"Schema validation error: {e}"
            short_err = str(e).splitlines()[0][:200]
            logging.warning(f"attempt={attempt} schema validation failed: {short_err}")

        if attempt == max_retries:
            break

        logging.info(f"retrying without re-sending images (attempt {attempt + 1}/{max_retries})")

        # Replay the model's own output as an assistant turn, with the
        # `thinking` sink stripped out of the JSON payload. `thinking=None`
        # on the message keeps ollama from re-injecting prior reasoning.
        try:
            resp_dict = json.loads(raw_response)
            resp_dict.pop("thinking", None)
            cleaned = json.dumps(resp_dict)
        except json.JSONDecodeError:
            cleaned = raw_response
        # Retry without re-sending the PDF page images. The model already
        # consumed them on attempt 0; re-prefilling 20 PNGs is the dominant
        # cost on hard documents.
        messages = [
            system_prompt_dict,
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": cleaned, "thinking": None},
            {"role": "user", "content": f"{error_preamble}\n\n{validation_error}"},
        ]

    if survey_obj is None:
        logging.error(f"extraction failed for {filename} after {attempt + 1} attempts")

    return ExtractionResult(
        filename=filename,
        survey=survey_obj,
        raw_response=raw_response,
        response_metrics=metrics,
        validation_error=validation_error,
        retry_count=attempt,
        success=survey_obj is not None,
        timestamp=datetime.now().isoformat(),
    )


def main():
    log_file_path = setup_logging()
    logging.info(f"logging to {log_file_path}")
    config = load_config(config_path="./config.yaml")
    logging.info(
        f"config: model={config.get('model')} input_dir={config.get('input_dir')} "
        f"output_path={config.get('output_path')} max_retries={config.get('max_retries')} "
        f"keep_alive={config.get('keep_alive')} seed={config.get('seed')}"
    )

    input_files = os.listdir(path=config['input_dir'])
    logging.info(f"found {len(input_files)} entries in {config['input_dir']}")
    random.seed(config['seed'])
    random.shuffle(input_files)
    filenames = [x for x in input_files if x.endswith('.pdf')]
    logging.info(f"{len(filenames)} PDFs queued (shuffled with seed={config['seed']})")

    records = []

    filenames = ["999941280_full_001.pdf"] # debug
    filenames = ["999979446_full_001.pdf"] # debug
    filenames = ["999971030_full_001.pdf"] # debug
    filenames = ["999973412_full_001.pdf"] # debug
    total = len(filenames)
    successes = 0
    failures = 0
    for idx, filename in enumerate(tqdm(filenames), start=1):
        logging.info(f"[{idx}/{total}] processing {filename}")
        pdf_path = os.path.join(config['input_dir'], filename)

        logging.info(f"sending document to model for extraction...")
        trace() # debug
        try:
            result = extract_survey(pdf_path, config)
        except Exception as e:
            logging.error(f"Error during model inference for file {filename}: {e}")
            failures += 1
            continue

        logging.info(f"received response from model for file {filename}")
        if result.validation_error:
            logging.error(f"{filename}: {result.validation_error}")
        if result.success:
            successes += 1
        else:
            failures += 1

        records.append({
            "filename": result.filename,
            "response": result.raw_response,
            "response_created_at": result.response_metrics["created_at"],
            "response_total_duration": result.response_metrics["total_duration"],
            "response_load_duration": result.response_metrics["load_duration"],
            "response_prompt_eval_duration": result.response_metrics["prompt_eval_duration"],
            "response_eval_duration": result.response_metrics["eval_duration"],
            "retry_count": result.retry_count,
            "success": result.success,
            "validation_error": result.validation_error,
            "timestamp": result.timestamp,
        })

        logging.info(f"completed extraction for file {filename}")
        trace() # debug
        try:
            pd.DataFrame(records).to_parquet(config["output_path"])
            logging.info(f"saved {len(records)} records to {config['output_path']}")
        except Exception as e:
            logging.error(f"Failed saving records as .parquet: {e}")

    logging.info(
        f"session complete: processed={total} successes={successes} failures={failures}"
    )


if __name__ == "__main__":
    main()
