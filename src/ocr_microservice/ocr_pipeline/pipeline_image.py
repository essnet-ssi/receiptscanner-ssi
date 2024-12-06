from PIL import Image

class PipelineImage:

    def __init__(self, image: Image.Image, filename: str):
        self.image = image
        self.filename = filename
