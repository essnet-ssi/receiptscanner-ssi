import glob
import os
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage
from pdf2image import convert_from_path

from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.config.default import Config, Pipeline, Models, Cache
from ocr_microservice.ocr_pipeline.ocr_pipeline import process
from ocr_microservice.ocr_pipeline.pre_ocr import pre_ocr


def get_data_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_dir = os.path.join(parent_dir, "data")
    return data_dir

def get_pdf_paths():
    data_dir = get_data_dir()
    data_dir = os.path.join(data_dir, "pdfs")
    data_dir = os.path.join(data_dir, "*")
    pdf_paths = glob.glob(data_dir)
    return pdf_paths


def test_pdf():
    # Arrange
    pdf_paths = get_pdf_paths()

    for _, path in enumerate(pdf_paths):
        images = convert_from_path(path)
        for index, image in enumerate(images):
            filename = f"{path.split('/')[-1]}_{index}"
            pipeline_image = PipelineImage(image, filename)       

            output_dir = os.path.join(get_data_dir(), "output")
            injector = Injector(Config(save_results=True, data_dir=output_dir),
                                Pipeline(), Models(), Cache())

            # Act
            results = process(pipeline_image, injector)

            # Assert
            assert isinstance(results, str)

