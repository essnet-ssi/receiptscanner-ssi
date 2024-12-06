# Training the YOLOS model

## Step 1: collect images

Collect images and divide into 3 subsets: train, test and validation. 'train' is used for training. 'test' is for testing yolos while it is being trained and 'validation' is for validating the final yolos output (not used by yolos itself).

Images need to be collected in: step1-collected-images/train, step1-collected-images/test, step1-collected-images/validation directories.

Except for the text on the receipts, images must be text-free as the next step uses text detection for its rotation algorithm.

## Step 2: rotate images

Rotate all images. This step is needed to make the crop of the receipt as tight as possible.

```bash
mkdir step2-rotated-images/{test,train,validation}
python rotate.py
```

## Step 3: annotate and create Coco output

- use a tool to add bounding boxes for object detection to the images (e.g. Label Studio)
- export in Coco format
- validation images should not be annotated because they are not being used in yolos training

The directory structure should look like:

```bash
step3-coco-formatted/train
step3-coco-formatted/train/result.json
step3-coco-formatted/train/images
step3-coco-formatted/train/images/<all-train-images>
step3-coco-formatted/test
step3-coco-formatted/test/result.json
step3-coco-formatted/test/images
step3-coco-formatted/test/images/<all-test-images>
```

## Step 4: train

- modify/adjust parameters in the main function in train.py

```bash
python train.py
```

## Step 5: test

```bash
python inference.py <model> <image>
E.g. python inference.py lightning_logs/version_2/checkpoints/epoch\=5-step\=252.ckpt step2-rotated-images/validation/rotated-Baksteen3-0.jpg
```
