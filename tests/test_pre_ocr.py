import os
from pathlib import Path
from PIL import Image

from ocr_microservice.ocr_pipeline.pre_ocr import pre_ocr
from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.config.default import Config, Pipeline, Models, Cache
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage


def get_data_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_dir = os.path.join(parent_dir, "data")
    return data_dir

def get_image(filename: str = "test_image_1.jpg"):
    data_dir = get_data_dir()
    image_path = os.path.join(data_dir, filename)
    image = Image.open(image_path)
    return image


def test_pre_ocr():
    # Arrange
    filename = "test_image_1.jpg"
    image = get_image(filename)

    output_dir = os.path.join(get_data_dir(), "output")
    injector = Injector(Config(save_results=True, data_dir=output_dir), Pipeline(), Models(), Cache())

    # Act
    results = pre_ocr.process(image, filename, injector)

    # Assert
    assert results is not None
    assert isinstance(results, Image.Image)