from __future__ import annotations
from PIL import Image
from datetime import datetime
from typing import TYPE_CHECKING

from ocr_microservice.ocr_pipeline.helpers.type_check import is_json
if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector


def save_file(filename: str, description: str, data, injector: Injector):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if is_json(str(data)):
        ext = "json"
    else:
        ext = "txt"

    filename = f"{injector.config.data_dir}/{timestamp}_{filename}_{description}.{ext}"
    with open(filename, "w") as outfile:
        outfile.write(str(data))


def save_image(filename: str, description: str, image: Image.Image, injector: Injector):
    if injector.config.save_results:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        image.save(f"{injector.config.data_dir}/{timestamp}_{filename}_{description}.jpg")
