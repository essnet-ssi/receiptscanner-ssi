from torch.utils.data import  Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class MaskedReceiptDataset(Dataset):
    def __init__(self, images, masked_images, transform):
        self.images = images
        self.masked_images = masked_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masked_images[index]

        if self.transform:
            image, mask = self.transform((image, mask))
        return image, mask
    
    def inspect_data(self, idx):
            image, mask = self.__getitem__(idx)
            pil_transform = transforms.ToPILImage()
            pil_image = pil_transform(image)
            overlay_image = pil_transform(mask)
            _, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(pil_image)
            ax[0].set_title("Original Image")
            ax[0].axis("off")
            ax[1].imshow(overlay_image)
            ax[1].set_title("Masked Image")
            ax[1].axis("off")
            plt.show()
