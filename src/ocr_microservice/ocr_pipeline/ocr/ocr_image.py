from __future__ import annotations
from PIL import Image
from typing import TYPE_CHECKING
import numpy as np

from ocr_microservice.ocr_pipeline.ocr.inspect_results import inspect_ocr

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector


def process(image: Image.Image, filename: str, injector: Injector) -> list:
    if injector.cache.ocr_data:
        ocr_result = injector.cache.ocr_data
    else:
        ocr_result = injector.models.paddle_model.ocr(np.asarray(image),cls=False)
    inspect_ocr(injector, image, filename, ocr_result)
    output = []
    for idx in range(len(ocr_result)):
        res = ocr_result[idx]
        for line in res:
            bbox = line[0]
            text, confidence = line[1]
            output.append({"ocr": (text, bbox, round(confidence, 3))})
    return output
