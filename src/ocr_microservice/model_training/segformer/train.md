# Receipt Segmentation Model Training

This documentation describes the process of training a segmentation model to identify and mask key regions on receipts using a SegFormer model. The data for training is obtained from labeled CSV files generated via a Label Studio project.

---

## Data Preparation

### CSV Labels

The dataset labels must be exported from Label Studio and can be downloaded from:

[Label Studio Dataset Export](https://ssi-wp3-label-studio.motusbuilder.io/projects/5/data)

Ensure the CSV file contains:
- `image`: paths to the images.
- `points`: polygon points in JSON format, labeling receipt regions.

Place the CSV file in your desired directory.

### Image Folder

Ensure all images referenced in the CSV file are downloaded and stored locally in the specified image folder.

## Code Overview

### Main Training Script (`train.py`)

This script loads images, applies transformations, sets up the model and training parameters, and initiates training.

#### Training Parameters:
- Batch size: 8
- Learning rate: `1e-4`
- Epochs: 500
- Loss Function: `BCEWithLogitsLoss` with positive weighting

## How to Train the Model

### Step-by-step Guide:

1. **Prepare the Environment**

   Install the required Python packages.
   ```bash
   pip install pandas torch torchvision matplotlib pillow tqdm
   ```

2. **Organize Data**

   - Place all images in the `images/` directory.
   - Ensure the CSV file with labels is downloaded and saved locally (e.g., `labels.csv`).

3. **Update Paths in `train.py`**

   ```python
   device = torch.device("cuda")  # Use GPU if available, otherwise CPU
   result_dir = './results'  # Directory for saving model weights
   image_dir = './images/'   # Directory containing original images
   label_csv = './labels.csv' # Path to downloaded CSV labels
   ```

3. **Start Training**

   Execute the script:
   ```bash
   python train.py
   ```

   The script will:
   - Load and transform data. 
   - Display data samples to verify correctness.
   - Train for 500 epochs, periodically saving model weights (`.pth`) every 25 epochs.

## Model Checkpoints

Model weights will be saved periodically every 25 epochs as:
```
{result_dir}/{device}_weights_segformer_{epoch}.pth
```

## Visualization

The dataset class includes an inspection method to visualize original and masked images, verifying data correctness before training.

---

## Customizing Training

- Modify epochs, batch size, learning rate, or device in `train.py` for your requirements.
- Adjust the transformation in `train_transform` for data augmentation strategies.

---


