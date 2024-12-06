import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

class RandomRotateTransform:

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image, mask = sample
        angle = np.random.uniform(-self.degrees, self.degrees)
        return TF.rotate(image, angle), TF.rotate(mask, angle)
    
class ComposeTransforms:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_mask):
        img, mask = img_mask
        for t in self.transforms:
            img, mask = t((img, mask))
        return img, mask
    
class ResizeTransform:

    def __init__(self, size):
        self.size = size

    def __call__(self, img_mask):
        image, mask = img_mask
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size)
        return image, mask
    
class RandomBrightnessTransform:

    def __call__(self, img_mask):
        image, mask = img_mask
        brightness_factor = np.random.uniform(0.7, 1.3)
        image = TF.adjust_brightness(image, brightness_factor)
        return image, mask


class ToTensorTransform:
    
    def __call__(self, img_mask):
        image, mask = img_mask
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

train_transform = ComposeTransforms([
    ResizeTransform((512, 512)),
    RandomRotateTransform(30),
    RandomBrightnessTransform(),
    ToTensorTransform(),
])

inference_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
