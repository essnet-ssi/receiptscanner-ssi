from ocr_microservice.ocr_pipeline.config.default import Models, HardwareType
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage
from ocr_microservice.ocr_pipeline.pre_ocr.helpers.rotate_with_paddle import rotate_image
from ocr_microservice.ocr_pipeline.injector import Injector
import os

input_directories =  ["step1-collected-images/train","step1-collected-images/test","step1-collected-images/validation"]
output_directories = ["step2-rotated-images/train",  "step2-rotated-images/test",  "step2-rotated-images/validation"]

batch_size = 100

if __name__ == '__main__':
    # paddle is required for image rotation
    models = Models(load_models=False)
    models._load_paddle_model(HardwareType.CONDA_GPU)
    injector = Injector(None, None, models, None)

    for input_dir,output_dir in zip(input_directories,output_directories):
        image_paths = [input_dir+"/"+f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith('.')]
        for image_path in image_paths:
            pipeline_image = PipelineImage(image_path)
            print(pipeline_image.path)
            rotated = rotate_image(pipeline_image, injector)
            rotated.save(f"{output_dir}/rotated-{pipeline_image.filename}")
    