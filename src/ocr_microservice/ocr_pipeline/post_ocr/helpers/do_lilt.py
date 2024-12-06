from __future__ import annotations

from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ImageProcessor,
)
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional
import torch
import numpy as np
from importlib import resources
from PIL import Image
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.resources import fonts

label2color = {
    "O": "lightyellow",
    "I-date_text": "darkgreen",
    "I-date_value": "darkgreen",
    "I-time_text": "darkgreen",
    "I-time_value": "darkgreen",
    "I-datetime": "darkgreen",
    "I-heading": "purple",
    "I-unused8": "",
    "I-unused7": "",
    "I-unused6": "",
    "I-tax.header": "darkviolet",
    "I-tax.description": "brown",
    "I-tax.price": "brown",
    "I-tax.price_excl": "brown",
    "I-tax.price_incl": "brown",
    "I-unused5": "",
    "I-unused4": "",
    "I-unused3": "",
    "I-unused2": "",
    "I-unused1": "",
    "I-store.name": "green",
    "I-store.address": "green",
    "I-store.phone": "green",
    "I-store.email": "green",
    "I-store.website": "green",
    "I-store.tax_id": "green",
    "I-store.unused3": "",
    "I-store.unused2": "",
    "I-store.unused1": "",
    "I-store.etc": "green",
    "I-item.header": "indigo",
    "I-item.quantity": "magenta",
    "I-item.description": "magenta",
    "I-item.unit_price": "magenta",
    "I-item.price": "magenta",
    "I-item.id": "magenta",
    "I-item.discount_description": "darkblue",
    "I-item.discount_price": "darkblue",
    "I-item.etc": "magenta",
    "I-item.unused11": "",
    "I-item.unused10": "",
    "I-item.unused9": "",
    "I-item.unused8": "",
    "I-item.unused7": "",
    "I-item.unused6": "",
    "I-item.unused5": "",
    "I-item.unused4": "",
    "I-item.unused3": "",
    "I-item.unused2": "",
    "I-item.unused1": "",
    "I-sub_total.text": "blue",
    "I-sub_total.price": "blue",
    "I-sub_total.discount_text": "blue",
    "I-sub_total.discount_price": "blue",
    "I-sub_total.discount_item_text": "blue",
    "I-sub_total.discount_item_price": "blue",
    "I-sub_total.tax_text": "blue",
    "I-sub_total.tax_price": "blue",
    "I-sub_total.tax_excl_text": "blue",
    "I-sub_total.tax_excl_price": "blue",
    "I-sub_total.tax_incl_text": "blue",
    "I-sub_total.tax_incl_price": "blue",
    "I-sub_total.service_text": "blue",
    "I-sub_total.service_price": "blue",
    "I-sub_total.item_rows_text": "blue",
    "I-sub_total.item_rows_value": "blue",
    "I-sub_total.quantity_text": "blue",
    "I-sub_total.quantity_value": "blue",
    "I-sub_total.etc_text": "blue",
    "I-sub_total.etc_price": "blue",
    "I-total.text": "darkred",
    "I-total.price": "darkred",
    "I-total.rounded_text": "darkred",
    "I-total.rounded_price": "darkred",
    "I-total.unused4": "",
    "I-total.unused3": "",
    "I-total.unused2": "",
    "I-total.unused1": "",
    "I-total.etc_text": "darkred",
    "I-total.etc_price": "darkred",
    "I-payment.cash_text": "darkblue",
    "I-payment.cash_price": "darkblue",
    "I-payment.change_text": "darkblue",
    "I-payment.change_price": "darkblue",
    "I-payment.other_text": "darkblue",
    "I-payment.other_price": "darkblue",
    "I-payment.details_total_text": "darkblue",
    "I-payment.details_total_price": "darkblue",
    "I-payment.etc_text": "darkblue",
    "I-payment.etc_price": "darkblue",
}


# draw results onto the image
def draw_boxes(image, final_result):
    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font_path = str(resources.path(fonts, "DejaVuSans.ttf"))
    font = ImageFont.truetype(font_path)
    for f in final_result:
        (word, box, confidence) = f["ocr"]
        scores = f["lilt"]
        # only take highest score
        (ner_index, ner, prob) = scores[0]
        draw.rectangle(box, outline=label2color[ner])
        text = ner + " (" + word + ") " + str(prob)
        draw.text(
            (box[0] + 10, box[1] - 10), text=text, fill=label2color[ner], font=font
        )
    return image


def normalize_bbox(b, width, height):
    ul, ur, dr, dl = b
    x1, y1 = ul
    x2, y2 = ur
    x3, y3 = dr
    x4, y4 = dl
    x_min = 1000 * (min(x1, x2, x3, x4) / width)
    y_min = 1000 * (min(y1, y2, y3, y4) / height)
    x_max = 1000 * (max(x1, x2, x3, x4) / width)
    y_max = 1000 * (max(y1, y2, y3, y4) / height)
    w = x_max - x_min
    h = y_max - y_min
    return [int(x_min), int(y_min), int(x_min + w), int(y_min + h)]


# helper function to unnormalize bboxes for drawing onto the image
def unnormalize_box(bbox, width, height):
    return [
        round(width * (bbox[0] / 1000)),
        round(height * (bbox[1] / 1000)),
        round(width * (bbox[2] / 1000)),
        round(height * (bbox[3] / 1000)),
    ]


def run_inference_chunk(image, words, boxes, confidences, model, processor):
    encoding = processor(
        image, words, boxes=boxes, truncation=False, max_length=512, return_tensors="pt"
    )
    del encoding["pixel_values"]

    ## create model input
    words.insert(0, "unused")
    confidences.insert(0, 0.0)

    # run inference
    outputs = model(**encoding)

    # for each bbox collect the logits
    l = 0
    bbox_prev = [-1, -1, -1, -1]
    logits = []
    results = []
    result_bboxes = []
    for bbox in encoding["bbox"][0]:
        bbox = bbox.tolist()

        logit = outputs.logits[0][l].tolist()

        if bbox == bbox_prev:
            logits.append(logit)
        else:
            if l != 0:
                results.append(np.mean(logits, 0))
            logits = [logit]
            result_bboxes.append(bbox)

        bbox_prev = bbox
        l += 1

    # last one is a tensor with bbox [0,0,0,0] -> DO NOT ADD IT
    # results.append(np.mean(logits,0))

    # get probabilities using softmax from logit score and convert it to numpy array
    probabilities_scores = [
        functional.softmax(torch.from_numpy(r), dim=-1).numpy() for r in results
    ]

    # get the position of the max value for each ocr text fragment
    # the position corresponds to the label (ner_tag)

    # sort but keep the index (~ner_tag/label)
    # only keep top 3 labels
    probabilities_scores_with_index = []
    i = 0
    for p_array in probabilities_scores:
        probabilities_scores_with_index.append(
            (
                words[i],
                sorted(list(enumerate(p_array)), key=lambda i: i[1], reverse=True)[:3],
                confidences[i],
            )
        )
        i += 1

    # for the final result, add the string of ner_tag (~index value) for convenience
    final_result = []
    i = 0
    width, height = image.size
    for t in probabilities_scores_with_index:
        (word, scores, confidence) = t
        new_scores = []
        for s in scores:
            (index, prob) = s
            new_scores.append((index, model.config.id2label[index], round(prob, 3)))
        final_result.append(
            {
                "ocr": (
                    word,
                    unnormalize_box(result_bboxes[i], width, height),
                    confidence,
                ),
                "lilt": new_scores,
            }
        )
        i += 1

    # remove the dummy 'unused' word
    final_result = final_result[1:]

    return final_result


# run inference
def run_inference(image, ocr_data, model, processor):
    ocr = {"words": [], "boxes": [], "confidences": []}
    width, height = image.size
    for o in ocr_data:
        word, box, confidence = o["ocr"]
        ocr["words"].append(word)
        ocr["boxes"].append(normalize_bbox(box, width, height))
        ocr["confidences"].append(confidence)

    words = ocr["words"]
    boxes = ocr["boxes"]
    confidences = ocr["confidences"]

    chunk_size = 10
    offset = 0

    final_result = []
    for i in range(1 + int(len(words) / chunk_size)):
        offset = i * chunk_size
        w = words[offset : offset + chunk_size]
        b = boxes[offset : offset + chunk_size]
        c = confidences[offset : offset + chunk_size]
        final_result.extend(run_inference_chunk(image, w, b, c, model, processor))

    if offset:
        w = words[offset:]
        b = boxes[offset:]
        c = confidences[offset:]
        final_result.extend(run_inference_chunk(image, w, b, c, model, processor))

    image = draw_boxes(image, final_result)

    return image, final_result


def run_lilt(image: Image.Image, data: dict, injector: Injector) -> Tuple[Image.Image, dict]:
    model = injector.models.lilt_model
    feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
    tokenizer = injector.models.lilt_auto_tokenizer
    # cannot use from_pretrained since the processor is not saved in the base model
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)

    image, result = run_inference(image, data, model, processor)
    return image, result
