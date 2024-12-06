import json
from tqdm import tqdm
from glob import glob
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

def _create_masked_image(image, points):
    width, height = image.size
    points = [(int(x[0] * width / 100), int(x[1] * height / 100)) for x in points]
    masked_image = Image.new('L', (width, height), 0)
    ImageDraw.Draw(masked_image).polygon(points, outline=255, fill=255)
    return masked_image
    
def load_data(image_folder, label_csv):
    original_images = []
    masked_images = []

    label_df = pd.read_csv(label_csv)
    for index, row in tqdm(label_df.iterrows(), total=label_df.shape[0], desc="Loading images"):
        original_image = ImageOps.exif_transpose(Image.open(image_folder + row['image'].split("/")[-1]))
        points = json.loads(row['label'])[0]['points']
        masked_image = _create_masked_image(original_image, points)
        original_images.append(original_image)
        masked_images.append(masked_image)
    
    return original_images, masked_images