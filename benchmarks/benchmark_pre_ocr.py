import os
from PIL import Image
from helpers import clear_previous_results, get_image_paths

from ocr_microservice.ocr_pipeline.pre_ocr import pre_ocr
from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.config.default import Config, Pipeline, Models, Cache
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage


if __name__ == "__main__":
    clear_previous_results("pre_ocr")
    image_paths = get_image_paths("input")
    injector = Injector(Config(save_results=True), Pipeline(), Models(), Cache())

    for path in image_paths:
        pipeline_image = PipelineImage(path)
        pre_processed_image = pre_ocr.process(pipeline_image, injector)

