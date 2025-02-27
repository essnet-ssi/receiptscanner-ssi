import argparse

from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.ocr_pipeline import process
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage
from PIL import Image, ImageOps
from pdf2image import convert_from_path
from pathlib import Path
import importlib

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=str, nargs='+', help="input image (multiple entries can be provided)")
    ap.add_argument("-u", "--user_selection", nargs='?', type=str,
                    help="user selection of where receipt is located in image e.g. '[(1025,499),(2307,673),(484,3185),(2107,3413)]'")
    ap.add_argument("-o", "--output_dir", type=str, help="output directory")
    ap.add_argument("-p", "--pipeline", type=str, default='default', help="pipeline")
    args = vars(ap.parse_args())

    pipeline = importlib.import_module("ocr_microservice.ocr_pipeline.config." + args['pipeline'])

    if args['output_dir'] == None:
        injector = Injector(pipeline.Config(save_results=True), pipeline.Pipeline(), pipeline.Models(), pipeline.Cache())
    else:
        injector = Injector(pipeline.Config(save_results=True,data_dir=args['output_dir']), pipeline.Pipeline(), pipeline.Models(), pipeline.Cache())

    image_data_list = []
    for image_path in args['images']:
        if image_path.lower().endswith('.pdf'):
            imgs = convert_from_path(image_path)
            for index, image in enumerate(imgs):
                filename = f"{image_path.split('/')[-1]}_{index}"
                image_data_list.append(PipelineImage(image, filename))
        else:
            image_data_list.append(PipelineImage(ImageOps.exif_transpose(Image.open(image_path).convert('RGB')),Path(image_path).name))

    process(image_data_list, injector)
    