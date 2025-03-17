# OCR Microservice

This project contains a microservice that extracts information from receipt images or PDF e-tickets.

## Table of Contents

- [Overview](#overview)
  - [Pre-OCR](#1-pre-ocr)
  - [OCR](#2-ocr)
  - [Post-OCR](#3-post-ocr)
- [Local Deployment](#local-deployment)
- [Usage](#usage)
- [Development](#development)
  - [Testing](#testing)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Authors](#authors)
- [License](#license)

## Overview

The core of the microservice is the OCR pipeline, which processes an image or PDF receipt through three steps:

1. **Pre-OCR**: Receipt detection and rotational correction.
2. **OCR**: Text detection and text recognition.
3. **Post-OCR**: Machine Learning-based and Rule-based logic for classifying OCR text.

The OCR pipeline runs in Python. It can be executed locally (see the [Usage](#usage) section) or deployed as a Kubernetes pod, interfacing through a RabbitMQ message bus.

### 1. Pre-OCR

The primary responsibilities of Pre-OCR include receipt detection, orientation correction, and image cropping. The input is either:

- A photo of a receipt, or
- An e-ticket (PDF), which is converted into a set of images

Two methods are implemented:

- Semantic segmentation
- Object detection

#### Semantic Segmentation

1. The Pre-OCR pipeline takes an image of a receipt as input.
2. It segments the image, labeling each pixel as part of the receipt or not.
3. The smallest possible bounding rectangle is drawn around the biggest 'island' of pixels classified as part of the receipt.
4. Heuristics are applied to ensure the receipt is oriented upright.
5. The receipt is cropped using the bounding rectangle.

##### Object detection

An alternative method is to use object detection for receipt detection. In this case a rectangular box is drawn around the receipt in the image.
The detection is very accurate but less fine-grained than semantic segmentation. Furthermore, it requires quite some processing to correct the receipt orientation.
The orientation is of the image is corrected by sequentially applying PaddleOCR to derive text orientation.

The used AI model for object detection is [YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos).

### 2. OCR

Several OCR models were compared:

- EasyOCR
- Tesseract
- PaddleOCR
- TrOCR
- TODO: (Anymore, Tim?)

A fine-tuned PaddleOCR model provided the best accuracy for this task. The model was trained on a small labeled dataset of ~300 receipts.

### 3. Post-OCR

The output of the OCR step includes text and text locations.
That information is used:

- to understand the receipt i.e. trying to give a meaning to the recognized text (by OCR),
- to correct OCR mistakes

#### Document understanding

In order to understand the receipts, the pipeline applies a fine-tuned model of [LiLT](https://huggingface.co/docs/transformers/main/model_doc/lilt).
See also [here](https://github.com/jpWang/LiLT).

LiLT combines text and layout (text position) information to label a text box. A variety of labels exist, ranging from store address to tax price, and can be found in *./src/ocr_microservice/model_training/lilt/labels.xlsx*.

#### Rule-Based Extraction

A rule-based correction step improves output quality by extracting:

- **Receipt date**: Identified using regular expression patterns on the OCR output.
- **Total price**: Detected by matching specific keywords (editable in [total_price_extractor.py](src/ocr_microservice/ocr_pipeline/rule_based_extractor/extractors/total_price_extractor.py)), which if found are then matched with strings that look like prices on the same line.
- **Shop name**: Detected similarly using keywords (editable in [shop_extractor.py](src/ocr_microservice/ocr_pipeline/rule_based_extractor/extractors/shop_extractor.py)).
- **Products and prices**: Identifies price columns and matches associated product names.

## Local Deployment

To run the microservice, you must first download trained ML models (not included in this repository). Contact the developers (see [Authors](#authors)) for these models. Place them in the [models](ocr_mircoservice/src/ocr_microservice/ocr_pipeline/resources/models) directory.

Instructions to set up a Conda environment are provided below. Two environment flavors are available: **CUDA-GPU** and **CPU** (Apple Silicon compatible).

After setting up the Conda environment, activate it and install the `ocr_microservice` package locally with:

```bash
pip install -e .
```
#### Conda **GPU-CUDA** Environment

References:

- [Pytorch getting started](https://pytorch.org/get-started/previous-versions/)
- [paddle ocr environment](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#how-to-check-your-environment)
- [paddle cor gpu](https://www.wheelodex.org/projects/paddlepaddle-gpu/)

```bash
conda create --name ocr_microservice_gpu_env python=3.10
conda activate ocr_microservice_gpu_env
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install transformers
conda install cudatoolkit=11.7
conda install evaluate
conda install tensorboard
conda install -c conda-forge pycocotools
conda install lightning -c conda-forge
pip install ~/Downloads/paddlepaddle_gpu-2.5.2-cp310-cp310-manylinux1_x86_64.whl # file can be downloaded https://www.wheelodex.org/projects/paddlepaddle-gpu/
pip install paddleocr
pip install 'transformers[torch]'
pip install segformer-pytorch
pip install pytest
pip install pika
pip install pdf2image
# https://github.com/PaddlePaddle/PaddleOCR/issues/10924
patch <HOME_DIR>/anaconda3/envs/ocr_microservice_cpu_env/lib/python3.10/site-packages/paddleocr/paddleocr.py < paddleocr.py.patch
```

Needed sometimes:

```bash
export LD_LIBRARY_PATH=/home/<username>/anaconda3/envs/ocr_microservice_gpu_env/lib/:$LD_LIBRARY_PATH
```

#### **CPU** environment

```bash
conda create --name ocr_microservice_cpu_env python=3.10
conda activate ocr_microservice_cpu_env
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
conda install transformers
conda install evaluate
conda install tensorboard
conda install -c conda-forge pycocotools
conda install lightning -c conda-forge
conda install pytorch-lightning -c conda-forge
pip install paddlepaddle==2.5.2
pip install paddleocr
pip install 'transformers[torch]'
pip install segformer-pytorch
pip install pytest
pip install pika
pip install pdf2image
# https://github.com/PaddlePaddle/PaddleOCR/issues/10924
# patch <HOME_DIR>/anaconda3/envs/ocr_microservice_cpu_env/lib/python3.10/site-packages/paddleocr/paddleocr.py < paddleocr.py.patch
```

## Usage

An example of how to run the microservice is available in [test_full_pipeline.py](tests/test_full_pipeline.py).  
The main entry point is the `process` function in [ocr_pipeline.py](src/ocr_microservice/ocr_pipeline/ocr_pipeline.py).

## Development

### Testing

After installing `pytest`, activate your environment and run:

```bash
pytest tests
```

## Model Training

- **Segformer semantic segmentation**: [train.md](src/ocr_microservice/model_training/segformer/train.md)
- **YOLOS object detection**: See *./src/ocr_microservice/model_training/yolos/README.md*
- **Paddle OCR text recognition**: See [paddleocr/README.md](src/ocr_microservice/model_training/paddleocr/README.md)
- **LiLT receipt understanding**: See *./src/ocr_microservice/model_training/lilt/README.md*

## Deployment

To deploy as a Kubernetes pod:

1. Build the Dockerfile.
2. Update Kubernetes configurations in [k8s](k8s).

## Authors

This project has been development with a grant from Eurostat.

Reference:

```bash
Project 101119594 — SSI — SMP-ESS-2022-TrustedSmartSurveys
```

The code was developed by CBS (Statistics Netherlands) and [hbits CV](hbits.io).
In alphabetical order, the main authors were:

- Pieter Beyens <pieter.beyens@hbits.io>
- Tom Oerlemans <ts.oerlemans@cbs.nl>
- Tim Schijvenaars <t.schijvenaars@cbs.nl>

## License

TODO I propose an MIT license
