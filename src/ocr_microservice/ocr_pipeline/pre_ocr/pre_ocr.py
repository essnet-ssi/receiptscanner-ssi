from __future__ import annotations
from PIL import Image
from typing import TYPE_CHECKING

from ocr_microservice.ocr_pipeline.pre_ocr.helpers.opencsv_helpers import find_min_area_rect, find_upright_angle, correct_rotation, crop_image
from ocr_microservice.ocr_pipeline.pre_ocr.helpers.segment_receipt import segment_receipt
from ocr_microservice.ocr_pipeline.helpers.save_file import save_image
from ocr_microservice.ocr_pipeline.pre_ocr.helpers.yolos_inference import process_file
from ocr_microservice.ocr_pipeline.pre_ocr.helpers.rotate_with_paddle import rotate_image, rotate_180_if_needed

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector

def receipt_segmentation(image: Image.Image, filename: str, injector: Injector) -> Image.Image:
    masked_array = segment_receipt(image, injector)
    contour_image, box_coordinates = find_min_area_rect(masked_array, injector)
    box_angle = find_upright_angle(box_coordinates)
    rotated_image, rotated_box_coordinates = correct_rotation(image, box_coordinates, box_angle, injector)
    cropped_image = crop_image(rotated_image, rotated_box_coordinates, injector)
    save_image(filename, '_contour_pre_ocr', contour_image, injector)
    save_image(filename, '_rotated_pre_ocr', rotated_image, injector)
    save_image(filename, '_cropped_pre_ocr', cropped_image, injector)
    return cropped_image

def receipt_object_detection_yolos(image: Image.Image, filename: str, injector: Injector) -> Image.Image:
    # rotate first so the crop later on is as close as possible against the receipt
    rotated_image = rotate_image(image, injector)
    # receipt detection
    contour_image, cropped_image = process_file(rotated_image, injector.models.yolos_model, injector.models.torch_device)
    # check up-down orientation
    rotated2_image = rotate_180_if_needed(cropped_image, injector)
    save_image(filename, "_rotated_pre_ocr", rotated_image, injector)
    save_image(filename, "_contour_pre_ocr", contour_image, injector)
    save_image(filename, "_cropped_pre_ocr", cropped_image, injector)
    save_image(filename, "_rotated2_pre_ocr", rotated2_image, injector)
    return rotated2_image

def process(image: Image.Image, filename: str, injector: Injector) -> Image.Image:
    if injector.pipeline.pre_ocr_algo == None:
        return image
    
    save_image(filename, '_original_pre_ocr', image, injector)
    pre_processed_image = injector.pipeline.pre_ocr_algo(image, filename, injector)
    save_image(filename, '_after_pre_ocr', pre_processed_image, injector)
    return pre_processed_image
