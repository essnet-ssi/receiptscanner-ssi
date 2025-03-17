import os
from PIL import Image

from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.config.default import Config, Pipeline, Models, Cache
from ocr_microservice.ocr_pipeline.ocr_pipeline import process
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage


def get_data_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_dir = os.path.join(parent_dir, "data")
    return data_dir

def get_image(filename: str):
    data_dir = get_data_dir()
    image_path = os.path.join(data_dir, filename)
    image = Image.open(image_path)
    return image

def test_full_pipeline():
    # Arrange
    filename = "test_image_1.jpg"
    image = get_image(filename)
    pipeline_image = PipelineImage(image, filename)
    
    output_dir = os.path.join(get_data_dir(), "output")
    injector = Injector(Config(save_results=True, data_dir=output_dir), Pipeline(), Models(), Cache())

    # Act
    extracted_receipt = process(pipeline_image, injector)

    # Assert
    assert extracted_receipt is not None
    assert isinstance(extracted_receipt, dict)
    assert 'products' in extracted_receipt
    assert 'receipt_date' in extracted_receipt
    assert 'shop_name' in extracted_receipt
    assert 'total_price' in extracted_receipt
    assert isinstance(extracted_receipt['products'], list)
    assert isinstance(extracted_receipt['receipt_date'], str)
    assert isinstance(extracted_receipt['shop_name'], str)
    assert isinstance(extracted_receipt['total_price'], float)
    assert len(extracted_receipt['products']) > 0
    assert extracted_receipt['receipt_date'] is not None
    assert extracted_receipt['shop_name'] is not None
    assert extracted_receipt['total_price'] is not None
    assert extracted_receipt['total_price'] > 0.0
