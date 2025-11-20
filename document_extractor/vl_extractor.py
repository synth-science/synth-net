import re
import time
import torch
import logging
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from common import timeout_method # type: ignore
from document import APAPsycTestDocument # type: ignore

logger = logging.getLogger('main')

class VisionLanguageModelExtractor():

    def __init__(self, model_path: str, timeout_seconds: int, device: str = "cuda"):

        self.model_path = model_path
        self.timeout_seconds = timeout_seconds
        self.device = device

        self.model = None
        self.processor = None

        self._load_model()

    def _load_model(self):

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def _generate(self, messages):

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        generation_config = {
            "max_new_tokens": 5120,
            "do_sample": False,
            "temperature": 1.0,
            "num_beams": 1,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_config)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

    def _clean_markdown_output(self, text):

        opening_pattern = r'^```(?:markdown)?\s*\n?'

        text = re.sub(opening_pattern, '', text, count=1, flags=re.IGNORECASE)

        closing_pattern = r'\n?```\s*$'

        text = re.sub(closing_pattern, '', text, count=1)

        return text.strip()

    @timeout_method
    def transcribe(self, document: APAPsycTestDocument):

        prompt = """
            Please extract all visible text from the image I've shared, using
            proper markdown formatting for text, headers, lists, tables, and
            text styles. Ensure UTF-8 encoding.
            Respond with valid markdown only, and nothing else.
        """

        result = {
            'transcript_duration_seconds': None,
            'transcript_content': None,
            'transcript_length': None,
            'transcript_has_error': False,
            'transcript_error': None,
        }
        start_time = time.time()
        try:
            outputs = []
            for i, image in enumerate(document.images):
                logger.info(f"Transcribing {i+1} / {len(document.images)} page(s)...")
                content = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt.strip()}
                ]
                messages = [{ "role": "user", "content": content}]
                output = self._generate(messages)
                clean_output = self._clean_markdown_output(output[0])
                outputs.append(clean_output)

            transcript = "\n".join(outputs)

            end_time = time.time()

            result['transcript_duration_seconds'] = end_time - start_time
            result['transcript_content'] = transcript
            result['transcript_length'] = len(transcript)

        except Exception as e:
            result['transcript_has_error'] = True
            result['transcript_error'] = str(e)

        return result