from __future__ import annotations
from PIL import Image
from typing import TYPE_CHECKING
from paddleocr import draw_ocr
from importlib import resources
import numpy as np
import io
import csv

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.resources import fonts
from ocr_microservice.ocr_pipeline.helpers.save_file import save_image, save_file


def inspect_ocr(injector: Injector, image: Image.Image, filename: str, ocr_result: np.ndarray):
    if injector.config.save_results:
        boxes = [line[0] for line in ocr_result[0]]
        txts = [line[1][0] for line in ocr_result[0]]
        scores = [line[1][1] for line in ocr_result[0]]
        font_path = str(resources.path(fonts, "arial.ttf"))
        ocr_result_image_array = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        ocr_result_image = Image.fromarray(ocr_result_image_array)

        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        for idx in range(len(ocr_result)):
            res = ocr_result[idx]
            for line in res:
                writer.writerow(line)

        save_file(filename, "_ocr", output.getvalue(), injector)
        save_image(filename, "_ocr", ocr_result_image, injector)
