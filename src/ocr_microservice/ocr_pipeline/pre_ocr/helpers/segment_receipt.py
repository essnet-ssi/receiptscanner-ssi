from __future__ import annotations
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.pre_ocr.helpers.data_transformers import inference_transform

def segment_receipt(image: Image.Image, injector: Injector) -> np.ndarray:
    transformed_image = inference_transform(image)
    transformed_image = transformed_image.unsqueeze(0).to(injector.models.torch_device)
    segmentation_model = injector.models.segformer_model
    with torch.no_grad(): mask_tensor = segmentation_model(transformed_image)
    resized_tensor = F.interpolate(mask_tensor, size=image.size[::-1], mode='bilinear', align_corners=False)
    mask_array = (resized_tensor > 0).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255
    return mask_array
