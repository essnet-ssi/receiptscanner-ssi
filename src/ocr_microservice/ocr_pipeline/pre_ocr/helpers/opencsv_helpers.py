from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
from PIL import Image
import numpy as np  
from matplotlib import pyplot as plt
from PIL import ImageDraw
import math
from datetime import datetime

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector

def find_upright_angle(box: np.ndarray, square_threshold: float=0.8) -> float:
    side_lengths = [
        np.linalg.norm(box[0] - box[1]),
        np.linalg.norm(box[1] - box[2]),
        np.linalg.norm(box[2] - box[3]),
        np.linalg.norm(box[3] - box[0])
    ]

    # Check if the box is approximately square
    if min(side_lengths) / max(side_lengths) > square_threshold:
        return 0    

    # Determine the coordinates of the longer sides
    idx_longest_side = np.argmax(side_lengths)
    p1, p2 = box[idx_longest_side], box[(idx_longest_side + 1) % 4]

    # Calculate the angle to vertical axis
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    # If the receipt is mostly on it's side, rotate 90 degrees to the right
    if -11 <= angle_deg <= 11:
        return -90 + angle_deg
    if 169 <= angle_deg <= 191:
        return 90 + angle_deg
    
    # Otherwise rotate the image such that the receipt is upright, with the least amount of rotation
    if -180 < angle_deg <= -90:
        return angle_deg + 90
    elif -90 < angle_deg <= 0:
        return angle_deg + 90
    elif 0 < angle_deg <= 90:
        return angle_deg - 90
    else:  # 90 < angle_deg <= 180
        return angle_deg - 90

def find_min_area_rect(img_array: np.ndarray, injector: Injector, padding: int = 0) -> (np.ndarray, float):
    contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    if padding > 0:
        center = np.mean(box, axis=0)
        for i in range(4):
            box[i] = box[i] + (box[i] - center) * (padding / np.linalg.norm(box[i] - center))

    contour_image = None
    if injector.config.save_results:
        img_array_color = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_array_color, [box], 0, (0, 255, 0), 20)
        contour_image = Image.fromarray(img_array_color)

    return contour_image, box


def correct_rotation(image: Image.Image, box_coordinates: np.ndarray, angle: float, injector: Injector) -> (Image.Image, np.ndarray):
    rotated_image = image.rotate(angle, expand=True)

    # Calculate a rotation matrix
    angle_rad = math.radians(-angle)
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)
    cx, cy = image.width / 2, image.height / 2
    rcx, rcy = rotated_image.width / 2, rotated_image.height / 2

    # Calculate coordinates after rotation
    rotated_coordinates = []
    for x, y in box_coordinates:
        tx, ty = x - cx, y - cy
        new_x = (cos_val * tx - sin_val * ty) + rcx
        new_y = (sin_val * tx + cos_val * ty) + rcy
        rotated_coordinates.append((new_x, new_y))

    if injector.config.save_results:
        draw = ImageDraw.Draw(rotated_image)
        for i in range(len(rotated_coordinates)):
            draw.line(rotated_coordinates[i] + rotated_coordinates[(i + 1) % len(rotated_coordinates)], fill=(255, 0, 0), width=5)

    return rotated_image, rotated_coordinates


def crop_image(image: Image.Image, box_coordinates: np.ndarray, injector: Injector) -> Image.Image:
    x_coordinates, y_coordinates = zip(*box_coordinates)
    min_x, max_x = min(x_coordinates), max(x_coordinates)
    min_y, max_y = min(y_coordinates), max(y_coordinates)
    min_x, min_y = max(min_x, 0), max(min_y, 0)
    max_x, max_y = min(max_x, image.width), min(max_y, image.height)
    cropped_image = image.crop((min_x, min_y, max_x, max_y))
    return cropped_image
