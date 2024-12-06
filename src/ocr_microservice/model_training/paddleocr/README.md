
# Paddle Training

This README contains pointers on how to train and export Detection and Recognition models for Paddle OCR for receipt scanning.

## Install

Follow the guidelines on the Github page of PaddleOCR for installing. Be sure to install everthing in a dedicated Conda environment, and make sure this environment is set up following the instructions on the PaddleOCR Github page. Use *paddlepaddle-gpu* to install paddle through pip for GPU users. Also make sure you have the right CUDA-version installed.

## Training

Training can be split into two (or three, depending if the CLS model also is counted) groups: **detection** model and the **recognition** model. The detection model _detects_ words in an image and binds them to an bounding box with annotation. The recognition model _recognizes_ shapes within that bounding box and tries to classify them (by 'assigning' meaning to them e.g. characters). The following specific steps have to be taken in order to train your own PaddleOCR models:

### 1. Convert the image data and annotations in a required format

For the de **Detector**, you need at least two .txt files as training and validation sets (make three, to also have an evaluation set). The format of these text-files is as follows (without spaces):

``image_name.ext \t [{"points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "transcription":text_annotation, {"points":[...]}] \n``

With each image a new line. The same amount of .txt files should be made when training the **Recognizer** (train, test, eval) with the following format (without spaces):

``image_name.ext \t Annotation \n``

### 2. Download models and sources form the PaddleOCR Website

These can be downloaded via the Github project page of PaddleOCR. You can use an existing, already trained, model for finetuning (look up their models such as **en_PP-OCRv3_rec**) or you can download other pre-trained models like the **MobileNet_V3_Large**. The available models for detectors and recognizers vary.

### 3. Change the .yml config files 

When cloning the PaddleOCR project, configs for training are already included. These can be found within the configs folder, and are seperated per type of training configs (folder _det_ for detector, folder _rec_ for regocnizer etc.). Within these folders, .yml files can be found for specific training routines. You can also create one of your own, but going with an existing config like _det_mv3_db.yml_ is easier. Change the following fields: *Global.pretrained_model* (path of the weights you have downloaded in step 2), *Train.dataset.data_dir* and *Train.dataset.label_file_list* (path of training images and training .txt-file respectively) and finally under *Eval.dataset.data_dir* and *Eval.dataset.label_file_list* (path of test images and test .txt-file). Other fields (like output directory, learning rate, loss-functions etc.) can also be set here.

### 4. Start training

To train, use the following commands:

    #Training
    !python tools/train.py -c config/det/det_mv3_db.yml \
    -o Global.pretrained_model=./pretrained/MobileNetV3_large_x0_5_pretrained

This the train-script recognizes wether it is a detector-training or recognizer-training. Do not forget to enable/disable gpu-training in the config. GPU/CPU training is set in the config.

### 5. Inference

The last step is to convert the model and export it to a usable format. This can be done through the following commands:

    !python tools/export_model.py -c config/det/det_mv3_db.yml -o
    Global.pretrained_model="./output/trained_model/best_accuracy"
    Global.save_inference_dir="./output/trained_inference/"

### 6. Use it

Load the weights when running PaddleOCR. This can be done by entering the *rec_model_dir* and *det_model_dir* when 
