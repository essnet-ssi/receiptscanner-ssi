from paddleocr import PaddleOCR,draw_ocr
from pathlib import Path
import csv
import argparse
from os import listdir
from os.path import isfile, join
import numpy as np
from ocr_microservice.ocr_pipeline.config.default import Models, HardwareType
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage
from PIL import Image, ImageOps

def process(args, img_path,writer,ocr_model):
    pipeline_image = PipelineImage(ImageOps.exif_transpose(Image.open(img_path).convert('RGB')),Path(img_path).name)
    result = ocr_model.ocr(np.asarray(pipeline_image.image), cls=False)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            text,confidence = line[1]
            line[1] = f"('{text}', " + f"%.3f)" % round(confidence, 3)
            if args['only_filename'] == True:
                line.insert(0,Path(img_path).name)
            else:
                line.insert(0,img_path)
            if args['append_ignore_label'] == True:
                line.append(0) # 'O' -> ignore by default (labels)
            else:
                line.append("")
            writer.writerow(line)

def main(args, ocr_model, directories):
    f = open(args['output'], 'w')
    writer = csv.writer(f,quoting=csv.QUOTE_ALL)

    for directory in directories:

        if isfile(directory):
            process(directory,writer)
            return

        images = [f for f in listdir(directory) if isfile(join(directory, f)) and not f.startswith('.')]
        images = sorted(images, key=str.casefold)

        for image in images:
            img_path = directory + "/" + image
            print(img_path)
            process(args,img_path,writer,ocr_model)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, help="dataset", default="my_dataset", required=False)
    ap.add_argument("--single_directory", nargs='?', help="dataset is not split into train/ and test/", const=True, required=False)
    ap.add_argument("--only_filename", nargs='?', help="only filename or full path in csv", const=True, required=False)
    ap.add_argument("--append_ignore_label", nargs='?', help="append ignore label to csv", const=True, required=False)
    ap.add_argument("--output", type=str, help="output file", default="my_dataset/data.csv", required=False)
    args = vars(ap.parse_args())

    # paddle is required for image rotation
    models = Models(load_models=False)
    models._load_paddle_model(HardwareType.CONDA_GPU)

    if args['single_directory'] == True:
        main(args, models.paddle_model, [f"{args['dataset']}"])
    else:
        main(args, models.paddle_model, [f"{args['dataset']}/train",f"{args['dataset']}/test"])
