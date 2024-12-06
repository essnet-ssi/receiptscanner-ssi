from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage

from ocr_microservice.ocr_pipeline.post_ocr.helpers.inspect_results import inspect_post_ocr_multi_image
if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector

def process(pipeline_images: List[PipelineImage], image_results: list, injector: Injector) -> str:
    combined_filename = '&'.join([image.filename for image in pipeline_images])
    images = [image for image in pipeline_images]
    for stage, action in enumerate(injector.pipeline.post_ocr_steps_multi_image):
        data = action(images, combined_filename, image_results, injector)
        inspect_post_ocr_multi_image(combined_filename, data, stage, injector)
    return data