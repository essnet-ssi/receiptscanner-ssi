from __future__ import annotations
from PIL import Image
from typing import TYPE_CHECKING
from typing import Union, List
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage


if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.should_pre_process import should_pre_process
from ocr_microservice.ocr_pipeline.pre_ocr import pre_ocr
from ocr_microservice.ocr_pipeline.ocr import ocr_image
from ocr_microservice.ocr_pipeline.post_ocr import post_ocr
from ocr_microservice.ocr_pipeline.multi_image import combine_multi_image


def _process_image(image: Image.Image, filename: str, injector: Injector, perform_pre_processing: bool) -> dict:
    perform_pre_processing = should_pre_process.process(image, filename, injector, perform_pre_processing)
    pre_processed_image = pre_ocr.process(image, filename, injector) if perform_pre_processing else image
    ocr_result = ocr_image.process(pre_processed_image, filename, injector)
    post_ocr_result = post_ocr.process(pre_processed_image, filename, ocr_result, injector)
    return post_ocr_result
    

def process(pipeline_images: Union[PipelineImage, List[PipelineImage]], injector: Injector, perform_pre_processing: bool = True) -> dict:
    """
    :param images: PipelineImage or list of PipelineImages (in case of a multi-image receipt PDF).
    :param injector: An object that contains configuration settings and models for the OCR pipeline.
    :param perform_pre_processing: A flag that can be used to skip pre-processing (in case of a PDF).
    """
    if not isinstance(pipeline_images, list):  pipeline_images = [pipeline_images]
    image_results = [_process_image(pipeline_image.image, pipeline_image.filename, injector, perform_pre_processing) for pipeline_image in pipeline_images]
    combined_result = combine_multi_image.process(pipeline_images, image_results, injector)
    return combined_result
    