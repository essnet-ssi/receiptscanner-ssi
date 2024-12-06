from __future__ import annotations
from PIL import Image
from ocr_microservice.ocr_pipeline.helpers.save_file import save_file, save_image
from typing import TYPE_CHECKING
if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector


def inspect_post_ocr(image: Image.Image, filename: str, stage: int, data: list, injector: Injector):
    if injector.config.save_results:
        save_image(filename, f"_post_ocr_{stage}", image, injector)
        save_file(filename, f"_post_ocr_{stage}", data, injector)


def inspect_post_ocr_multi_image(filename: str, data: list, stage: int, injector: Injector):
    if injector.config.save_results:
        save_file(filename, f"_post_ocr_multi_{stage}", data, injector)
