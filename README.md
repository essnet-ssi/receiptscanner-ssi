# OCR Microservice

This project contains a microservice that extracts information from receipt images or pdf e-tickets using OCR and outputs it as a json file.

## Table of Contents

- [Usage](#usage)
- [Overview](#overview)
  - [Pre-OCR](#1-pre-ocr)
  - [OCR](#2-ocr)
  - [Post-OCR](#3-post-ocr)
  - [Integration](#4-integration)
- [Development](#development)
  - [Getting Started](#getting-started)
  - [Testing](#testing)
  - [Design and implementation](#design-and-implementation)
  - [Benchmarking](#benchmarking)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Authors](#authors)
- [License](#license)

## Usage

TODO all
TODO where to put run.py?

### python

## Overview

The core of the microservice is the OCR pipeline which takes an image or (pdf) e-ticket through 3 steps:

- Pre-OCR: preprocessing the received image or pdf
- OCR: optical character recognition
- Post-OCR
  - document (receipt) understanding, and
  - final post-processing which output a json file.

The OCR pipeline is a python runtime. It can be deployed as a microservice (Docker container). Communication with this microservice is done via RabbitMQ.
The python runtime can also be integrated in other ways (not available in this git repo).

### 1. Pre-OCR

Main responsibilities of Pre-OCR are receipt detection, orientation correction and image cropping.
The input is a single receipt in the form of:

- a set of images, or
- an e-ticket (pdf). The pdf is converted into a set of images which then follows the same flow as normal images.

Two methods have been deployed:

- semantic segmentation, and
- object detection

#### Semanantic segmentation

1. The pre-ocr pipeline takes as input an image of a receipt.
2. It then segments the image. Which means that it labels each pixel of the receipt as belonging to the receipt or not.
3. It then draws the smallest possible bounding rectangle around the pixels classified as belonging to the receipt.
4. It uses heuristics to try and correctly orrient the receipt, such that it is upright.
5. It crops the receipt using the bounding rectangle.
6. It then adjusts the orientation again slightly, using paddle_ocr to fine tune the orientation.

##### Object detection

An alternative method is to use object detection for receipt detection. In this case a rectangular box is drawn around the receipt in the image.
The detection is very accurate but less fine-grained than semantic segmentation. Furthermore, it requires quite some processing to correct the receipt orientation.
The orientation is of the image is corrected by sequentially applying PaddleOCR to derive text orientation.

The used AI model for object detection is [YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos).

### 2. OCR

TODO Tim

### 3. Post-OCR

The output of the OCR step includes text and text locations.
That information is used:

- to understand the receipt i.e. trying to give a meaning to the recognized text (by OCR),
- to correct OCR mistakes (TODO Tim - date, time corrections etc.),
- to produce a final json output which contains all receipt details (and some metadata).

#### Receipt understanding

In order to understand the receipts, the pipeline applies a fine-tuned model of [LiLT](https://huggingface.co/docs/transformers/main/model_doc/lilt).
See also [here](https://github.com/jpWang/LiLT).

LiLT combines text and layout (text position) information to label a text box. A variety of labels exist, ranging from store address to tax price, and can be found in *./src/ocr_microservice/model_training/lilt/labels.xlsx*.

#### Corrections

TODO Tim

#### Final post-processing

Given the output of receipt understanding and its corrections, a json output is being generated which contains all found information as well as metadata (i.e. where the information came from).


## Development

### Getting started

To setup a local development environment, you first need to setup a Conda environment. The following instructions provide guidance on creating this environment for a GPU that supports CUDA, as well as for a CPU setup, which is also compatible with the latest Apple Silicon chips.

After setting up the Conda environment, activate it, and then install the ocr_microservice package locally. Run the following command in the root folder of this project, where `pyproject.toml` is located:

```bash
pip install -e .
```

Lastly, download the machine learning models (which are not included in the git repository) and place them inside `./src/ocr_microservice/ocr_pipeline/resources/models`.

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
pip install paddlepaddle=2.5.2
pip install paddleocr
pip install 'transformers[torch]'
pip install segformer-pytorch
pip install pytest
pip install pika
pip install pdf2image
# https://github.com/PaddlePaddle/PaddleOCR/issues/10924
patch <HOME_DIR>/anaconda3/envs/ocr_microservice_cpu_env/lib/python3.10/site-packages/paddleocr/paddleocr.py < paddleocr.py.patch
```

### Testing

After installing pytest inside your Conda environment, you may need to reactivate it.

Then to run the tests, run the following command inside the root directory:

```bash
pytest tests
```

### Design and implementation

This OCR microservice is organized for modular, scalable processing using RabbitMQ and Kubernetes.

#### Key Components

- **`message_bus_consumer/`**  
  Listens to a RabbitMQ queue for image filenames, runs OCR, and publishes structured results back.

- **`ocr_pipeline/`**  
  Core OCR logic with:
  - `pre_ocr/`, `post_ocr/`: Image preparation and result refinement  
  - `ocr_pipeline.py`: Main processing flow  
  - `config/`, `injector.py`: Configuration and dependency injection  
  - `resources/`: Fonts and assets

- **`model_training/`**  
  Training scripts for Lilt, PaddleOCR, Segformer, and YOLOS models with dataset folders.

#### Build & Setup

- Uses `pyproject.toml` with `src/` layout (`package-dir = {"" = "src"}`)
- Static assets (e.g., fonts) included via `package-data`
- Designed for container deployment with RabbitMQ and image volume mounting

### Benchmarking

To evaluate the OCR pipeline, a benchmarking script is provided in benchmarks/run_benchmark.py. This script processes a set of annotated receipt images and compares the extracted data to ground-truth labels.

To run this benchmarking, make sure your environment is correctly set up:

Place your annotated benchmark data (images and annotation.json) inside benchmarks/assets/.

The annotation.json should follow this structure:

```{
  "test_image_1.JPG": {
    "date": "2023-01-01",
    "articles": [ ... ],
    "total": 12.34
  },
  ...
}
```

And finally run python benchmarks/run_benchmark.py

## Model Training

### receipt detection: segformer semantic segmentation

Segformer training is described in [train.md](src/ocr_microservice/model_training/segformer/train.md).

### receipt detection: yolos object detection

See the description in *./src/ocr_microservice/model_training/yolos/README.md*.

### paddle OCR text recognition

TODO Tim

### lilt receipt understanding

See the description in *./src/ocr_microservice/model_training/lilt/README.md*.

## Deployment

To deploy this service in a Kubernetes cluster:

1.  Run the following command from the root of the project:
`docker build -t <your-docker-image-name> .`
2. Update Kubernetes manifests
Modify the configuration files in the k8s directory:
    * Set the correct Docker image name in the deployment YAML.
    * Provide the necessary RabbitMQ environment variables: `RABBITMQ_HOST` = the hostname of your RabbitMQ instance.
    * Mount the image folder used to store the receipt images. This should be configured as a volume, and mapped to `/mnt/images` inside the container.
    * Set other preferences like the number of replicas.
3. Apply the Kubernetes configs
`kubectl apply -f k8s/`
4. How it works:
The `message_bus_consumer.py` file runs the main backend service. It listens for image filenames on a RabbitMQ queue (`receipt_requests_queue`), processes the image using the OCR pipeline, and publishes the result to another queue (`receipt_results_queue`).

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
- Tim Schijvenaars TODO Tim

## License

See LICENSE.md
