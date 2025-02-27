from __future__ import annotations
from transformers import LiltForTokenClassification, LayoutLMv3Processor, LayoutLMv3ImageProcessor, AutoTokenizer
from pathlib import Path
import json
from typing import TYPE_CHECKING
from PIL import Image
from typing import List

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.post_ocr.helpers.label_groups import LabelGroups

def to_json(images: List[Image.Image], filename: str, image_results: list, injector: Injector):
    if len(image_results):
        return json.dumps(image_results[0], indent=2)
    else:
        return json.dumps(image_results, indent=2)
