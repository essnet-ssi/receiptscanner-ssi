# Training LiLT

## Step 1: collect receipts and apply ocr

Collect receipts in 3 sets:

- train: for training the LiLT model, directory: 'my_dataset/train'
- test: for testing the LiLT model while it is being trained, directory: 'my_dataset/test'
- validation: for validation after the LiLT has been trained. The validation set does not need to be annotated (except if needed for automatic regression testing), directory: 'my_dataset/validation'

Images need to be correctly oriented and cropped.

Then, use the script 'paddleocr2csv.py' to apply ocr:

```bash
python paddleocr2csv.py
```

The output is a data file:

```bash
my_dataset/data.csv
```

## Step 2: annotate and compose the dataset

- annotate 'data.csv' with the labels shown in labels.xlsx. Use the numeric values.

## Step 3: train

Start training with the coomand

```bash
python train.py my_dataset
```

The model will be written to the 'my_dataset-model' directory.

## Step 4: copy some extra files to the model

```bash
cd my_dataset-model
cp ../extra-model-files/* .
```

## Step 5: test inference

```bash
python inference <image>
```

Output is printed and written to 'inference.jpg'.
