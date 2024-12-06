from __future__ import annotations
import numpy as np
from PIL import Image
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector

# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle(bbox):
    ul,ur,dr,dl = bbox
    x1,y1 = ul
    x2,y2 = ur

    x_min = min(x1,x2)
    x_max = max(x1,x2)
    y_min = min(y1,y2)
    y_max = max(y1,y2)

    p0 = [x_min, y_min]
    p1 = [x_max, y_min]
    p2 = [x_min, y_max]

    ''' 
    compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    degrees = round(np.degrees(angle),2)

    # bbox is slanted in one way or the other
    if y1 > y2:
        degrees *= -1
    
    return degrees

def is_rectangular_enough(bbox):
    ul,ur,dr,dl = bbox
    x1,y1 = ul
    x2,y2 = ur
    x3,y3 = dr
    x4,y4 = dl

    w = min(abs(x2-x1),abs(x4-x3))
    h = min(abs(y3-y1),abs(y4-y2))

    if w == 0 or h == 0:
        return False

    if min(w,h)/max(w,h) < 0.5:
        return True
    else:
        return False

def is_90degrees_rotated(bboxes):
    count = 0
    for bbox in bboxes:
        ul,ur,dr,dl = bbox
        x1,y1 = ul
        x2,y2 = ur
        x3,y3 = dr
        x4,y4 = dl

        if abs(x2 - x1) < abs(y3-y2):
            count += 1
    
    if count > (len(bboxes)/2):
        return True
    else:
        return False

def rotate(image: Image, injector: Injector):
    result = injector.models.paddle_model.ocr(np.asarray(image),rec=False,cls=False)

    # get bboxes
    bboxes = []
    for idx in range(len(result)):
        res = result[idx]
        if res == None:
            return None
        for line in res:
            bboxes.append(line)
    
    is_90d_rotated = is_90degrees_rotated(bboxes)

    # get angles
    angles = []
    for bbox in bboxes:
        # avoid square bounding boxes
        if is_rectangular_enough(bbox):
            angles.append(angle(bbox))
    
    rotation = statistics.mean(angles)
    correction = -1*rotation
    if is_90d_rotated:
        correction += 90

    image = image.rotate(correction,expand=True)

    return image

def text_confidences(image: Image, injector: Injector):
    ocr_result = injector.models.paddle_model.ocr(np.asarray(image),cls=False)

    confidences = []
    for idx in range(len(ocr_result)):
        res = ocr_result[idx]
        if res == None:
            return None
        for line in res:
            (_,(text,confidence)) = line
            if len(text) >= 4: # important to only take into account significant text
                confidences.append(confidence)
    
    return confidences,ocr_result

def rotate_image(image: Image.Image, injector: Injector):
    # rotate using paddle -> only text detection (no recognition)
    image = rotate(image,injector)
    image = rotate(image,injector)

    return image

def rotate_180_if_needed(image: Image.Image, injector: Injector):
    # fine tuning and get text confidences
    confidences1,ocr_result = text_confidences(image,injector)
    injector.cache.ocr_data = ocr_result

    # rotate 180 degrees and check whether text confidence becomes better
    image180 = image.rotate(180,expand=True)
    confidences2,ocr_result_180 = text_confidences(image180,injector)
    mean1 = 0
    if confidences1 != None and len(confidences1):
        mean1 = statistics.mean(confidences1)
    mean2 = 0
    if confidences2 != None and len(confidences2):
        mean2 = statistics.mean(confidences2)
    if mean1 < mean2:
        image = image180
        injector.cache.ocr_data = ocr_result_180

    return image
