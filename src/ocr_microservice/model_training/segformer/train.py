#%%
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ocr_microservice.ocr_pipeline.pre_ocr.helpers.data_transformers import train_transform
from ocr_microservice.model_training.segformer.helpers.data_loader import load_data
from ocr_microservice.model_training.segformer.helpers.masked_receipt_dataset import MaskedReceiptDataset
from ocr_microservice.ocr_pipeline.pre_ocr.helpers.segmentation_model import segmentation_model


def train(image_folder: str, label_csv_path: str, result_dir: str, device: torch.device) -> None:
    # Load data
    original_images, masked_images = load_data(image_folder, label_csv_path)
    masked_receipt_dataset = MaskedReceiptDataset(original_images, masked_images, train_transform)
    print('Showing examples of loaded data.')
    for i in range(3): masked_receipt_dataset.inspect_data(i)
    dataloader = DataLoader(masked_receipt_dataset, batch_size=8, shuffle=True)

    # Define model and training parameters
    segmentation_model.to(device)
    receipt_weight = 3  # decreases the chance of false negatives, and increases the chance of false positives - because misclassyfing a receipt pixel as a non-receipt is worse than the other way around
    weights = torch.tensor([receipt_weight], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=1e-4)
    num_epochs = 500

    # Train model
    print('Starting training')
    for epoch in range(0, num_epochs):
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = segmentation_model(images)
            outputs = F.interpolate(outputs, masks.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        correct_predictions = (torch.sigmoid(outputs) > 0.5) == masks
        accuracy = correct_predictions.sum().item() / (masks.size(0) * masks.size(1) * masks.size(2) * masks.size(3))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {time.ctime()}")

        if epoch % 25 == 0:
            torch.save(segmentation_model.state_dict(), f'{result_dir}/{device}_weights_segformer_{epoch}.pth')


if __name__ == "__main__":
    device = torch.device("cpu")  # options: cpu/cuda/mps
    result_dir = '' # where to save the model weights
    image_dir = '' # where the images are stored
    label_csv_path = '' # where the labels are stored
    train(image_dir, label_csv_path, result_dir, device)

# %%
