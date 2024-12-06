import os
from PIL import Image
from helpers import clear_previous_results, get_image_paths

from pathlib import Path
import argparse


from ocr_microservice.ocr_pipeline.pre_ocr import pre_ocr
from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.config.default import Config, Pipeline, Models, Cache
from ocr_microservice.ocr_pipeline.ocr_pipeline import process


if __name__ == "__main__":
    clear_previous_results("pre_ocr")
    clear_previous_results("ocr")
    clear_previous_results("post_ocr")

    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=str, nargs='+', help="input image (multiple entries can be provided)")
    args = vars(ap.parse_args())

    image_paths = args['images']
    injector = Injector(Config(save_results=True, data_dir="../data"), Pipeline(), Models(), Cache())

    for path in image_paths:
        image = Image.open(path)
        filename = Path(path).stem
        results = process(image, filename, injector)




    