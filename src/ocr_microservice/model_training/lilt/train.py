import argparse
import os
from datasets import load_dataset, Features, Sequence, ClassLabel, Value, Array2D
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv3ImageProcessor, AutoTokenizer, LayoutLMv3Processor, LiltForTokenClassification, Trainer, TrainingArguments
from PIL import Image
from functools import partial
from huggingface_hub import HfFolder
import evaluate
import numpy as np


# preprocess function to prepare into the correct format for the model
def preprocess_dataset(sample, processor=None):
    for bbox in sample["bboxes"]:
        x1, y1, x2, y2 = bbox
        if (x2-x1 < 1) or (y2-y1 < 1):
            print("BUG BBOX: ZERO FOUND !!!")
        if x1 > 999 or y1 > 999 or x2 > 999 or y2 > 999:
            print("BUG BBOX: >=1000 !!!")
            print(bbox)
    encoding = processor(
        Image.open(sample["image_path"]).convert("RGB"),
        sample["words"],
        boxes=sample["bboxes"],
        word_labels=sample["ner_tags"],
        padding="max_length",
        truncation=True,
    )
    # remove pixel values not needed for LiLT
    del encoding["pixel_values"]
    return encoding

class ComputeMetrics(object):
    def __init__(self, metric, ner_labels):
        self.metric = metric
        self.ner_labels = ner_labels
    
    def __call__(self,p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        all_predictions = []
        all_labels = []
        for prediction, label in zip(predictions, labels):
            for predicted_idx, label_idx in zip(prediction, label):
                if label_idx == -100:
                    continue
                all_predictions.append(self.ner_labels[predicted_idx])
                all_labels.append(self.ner_labels[label_idx])
        return self.metric.compute(predictions=[all_predictions], references=[all_labels])

def main(args):
    dataset = load_dataset(args["dataset"])

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    labels = dataset['train'].features['ner_tags'].feature.names
    print(f"Available labels: {labels}")

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    model_id="SCUT-DLVCLab/lilt-roberta-en-base"

    # use LayoutLMv3 processor without ocr since the dataset already includes the ocr text
    feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False) # set 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # cannot use from_pretrained since the processor is not saved in the base model
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)

    # we need to define custom features
    features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(feature=Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "labels": Sequence(ClassLabel(names=labels)),
        }
    )

    # process the dataset and format it to pytorch
    proc_dataset = dataset.map(
        partial(preprocess_dataset, processor=processor),
        #remove_columns=["image", "tokens", "ner_tags", "id", "bboxes"], # CORD
        remove_columns=["id", "words", "image_path", "ner_tags", "bboxes"],
        features=features,
    ).with_format("torch")

    print(proc_dataset["train"].features.keys())

    # huggingface hub model id
    model_id = "SCUT-DLVCLab/lilt-roberta-en-base"

    # load model with correct number of labels and mapping
    model = LiltForTokenClassification.from_pretrained(
        model_id, num_labels=len(labels), label2id=label2id, id2label=id2label
    )

    # load seqeval metric
    metric = evaluate.load("seqeval")

    # labels of the model
    ner_labels = list(model.config.id2label.values())

    # hugging face parameter
    repository_id = args["dataset"] + "-model"

    # Define training args
    training_args = TrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=True,
        learning_rate=args["learning_rate"],
        max_steps=args["max_steps"],
        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=proc_dataset["train"],
        eval_dataset=proc_dataset["test"],
        compute_metrics=ComputeMetrics(metric, ner_labels),
    )

    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str, help="dataset")
    ap.add_argument("--max_steps", type=int, help="max number of steps", default=5000, required=False)
    ap.add_argument("--learning_rate", type=int, help="learning rate", default=5e-5, required=False)
    args = vars(ap.parse_args())

    main(args)

