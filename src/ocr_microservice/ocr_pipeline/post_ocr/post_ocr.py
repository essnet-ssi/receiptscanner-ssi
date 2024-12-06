from __future__ import annotations
from PIL import Image
from typing import TYPE_CHECKING

from ocr_microservice.ocr_pipeline.post_ocr.helpers.inspect_results import inspect_post_ocr
if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector

def process(orig_image: Image.Image, filename: str, data: list, injector: Injector) -> dict:
    for stage, action in enumerate(injector.pipeline.post_ocr_steps):
        image, data = action(orig_image.copy(), data, injector)
        inspect_post_ocr(image, filename, stage, data, injector)
    return data