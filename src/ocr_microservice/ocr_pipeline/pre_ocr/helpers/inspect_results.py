from __future__ import annotations
from PIL import Image
from typing import TYPE_CHECKING

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector


def inspect_pre_ocr(image: Image.Image, injector: Injector):
    if injector.config.save_results:
        image.save(f"{injector.config.data_dir}/pre_ocr/{injector.filename}.jpg")
