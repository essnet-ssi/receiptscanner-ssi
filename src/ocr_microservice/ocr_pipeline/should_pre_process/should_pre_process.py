from __future__ import annotations
from PIL import Image
from typing import TYPE_CHECKING
from ocr_microservice.ocr_pipeline.should_pre_process.helpers.count_unique_colors import count_unique_colors

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector


def process(image: Image.Image, filename: str, injector: Injector, perform_pre_processing: bool) -> bool:
    if perform_pre_processing == False: return False
    if '.pdf' in filename: return False
    unqiue_colors = count_unique_colors(image)
    if unqiue_colors < 100: return False

    # TODO: test if these are good conditions
    # if image.getexif() == {}: return False 
    # if 1650 <= image.size[0] <= 1660 and 2330 <= image.size[1] <= 2345: return False
    return True

