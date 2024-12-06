from transformers import LiltForTokenClassification, LayoutLMv3Processor, LayoutLMv3ImageProcessor, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
import argparse
from torch.nn import functional
import torch
from ocr_microservice.ocr_pipeline.config.default import Models, HardwareType
from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage
from ocr_microservice.ocr_pipeline.injector import Injector
import numpy as np

# helper function to unnormalize bboxes for drawing onto the image
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

label2color = {
    "O": "",
    "I-date_text": "darkgreen",
    "I-date_value": "darkgreen",
    "I-time_text": "darkgreen",
    "I-time_value": "darkgreen",
    "I-datetime": "darkgreen",
    "I-heading": "purple",
    "I-unused8": "",
    "I-unused7": "",
    "I-unused6": "",
    "I-tax.header": "brown",
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
    "I-item.quantity": "black",
    "I-item.description": "black",
    "I-item.unit_price": "black",
    "I-item.price": "black",
    "I-item.id": "black",
    "I-item.discount_description": "darkblue",
    "I-item.discount_price": "darkblue",
    "I-item.etc": "black",
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
    "I-payment.etc_price": "darkblue"
}

# draw results onto the image
def draw_boxes(image, final_result):
    width, height = image.size

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('fonts/DejaVuSans.ttf')
    for (word, scores, box) in final_result:
        unnormalized_box = unnormalize_box(box, width, height)
        # only take highest score
        (ner_index, ner, prob) = scores[0]
        if ner_index == 0: # ignore
            continue
        draw.rectangle(unnormalized_box, outline=label2color[ner])
        text = ner + ' ('+word+') ' + str(prob)
        draw.text((unnormalized_box[0] + 10, unnormalized_box[1] - 10), text=text, fill=label2color[ner], font=font)
    return image

def normalize_bbox(b,width,height):
    ul,ur,dr,dl = b
    x1,y1 = ul
    x2,y2 = ur
    x3,y3 = dr
    x4,y4 = dl
    x_min = 1000 * (min(x1,x2,x3,x4)/width)
    y_min = 1000 * (min(y1,y2,y3,y4)/height)
    x_max = 1000 * (max(x1,x2,x3,x4)/width)
    y_max = 1000 * (max(y1,y2,y3,y4)/height)
    w = x_max - x_min
    h = y_max - y_min
    return [int(x_min),int(y_min),int(x_min+w),int(y_min+h)]

def run_paddleocr(img_path, injector: Injector) :
    output = {
        "boxes": [],
        "words": []
    }
    im = Image.open(img_path)
    width, height = im.size
    result = injector.models.paddle_model.ocr(np.asarray(im), cls=False)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            bbox = line[0]
            text,confidence = line[1]
            output["boxes"].append(normalize_bbox(bbox,width,height))
            output["words"].append(text)
    return output

# run inference
def run_inference(image_path, ocr, model, processor, out_path):
    image = Image.open(image_path).convert('RGB')

    encoding = processor(image, ocr["words"], boxes=ocr["boxes"], truncation=True, max_length=512, return_tensors="pt")
    del encoding["pixel_values"]

    ## create model input
    words = ocr["words"]
    words.insert(0,"unused")

    # run inference
    outputs = model(**encoding)

    # for each bbox collect the logits
    import numpy as np
    l = 0
    bbox_prev = [-1,-1,-1,-1]
    logits = []
    results = []
    result_bboxes = []
    for bbox in encoding['bbox'][0]:
        bbox = bbox.tolist()

        logit = outputs.logits[0][l].tolist()

        if bbox == bbox_prev:
            logits.append(logit)
        else:
            if l != 0:
                results.append(np.mean(logits,0))
            logits = [logit]
            result_bboxes.append(bbox)

        bbox_prev = bbox
        l += 1

    # last one is a tensor with bbox [0,0,0,0] -> do not add it
    #results.append(np.mean(logits,0))

    # get probabilities using softmax from logit score and convert it to numpy array
    probabilities_scores = [functional.softmax(torch.from_numpy(r), dim = -1).numpy() for r in results]

    # get the position of the max value for each ocr text fragment
    # the position corresponds to the label (ner_tag)

    # sort but keep the index (~ner_tag/label)
    # only keep top 3 labels
    probabilities_scores_with_index = []
    i = 0
    for p_array in probabilities_scores:
        probabilities_scores_with_index.append((words[i],sorted(list(enumerate(p_array)), key=lambda i: i[1],reverse=True)[:3]))
        i += 1

    # for the final result, add the string of ner_tag (~index value) for convenience
    final_result = []
    i = 0
    for t in probabilities_scores_with_index:
        (word,scores) = t
        new_scores = []
        for s in scores:
            (index,prob) = s
            new_scores.append((index,model.config.id2label[index],round(prob,3)))
        final_result.append((word,new_scores,result_bboxes[i]))
        i += 1

    # remove the dummy 'unused' word
    final_result = final_result[1:]

    if out_path:
        draw_boxes(image,final_result).save(out_path)

    return final_result

def main(img_path, injector: Injector):
    print(img_path)

    # load model and processor
    model = LiltForTokenClassification.from_pretrained("my_dataset-model/")
    feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
    tokenizer = AutoTokenizer.from_pretrained("my_dataset-model/")
    # cannot use from_pretrained since the processor is not saved in the base model
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)

    # OCR
    ocr = run_paddleocr(img_path, injector)

    # INFERENCE (image + OCR)
    result = run_inference(img_path, ocr, model, processor, out_path="inference.jpg")

    print(result)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str, help="image")
    args = vars(ap.parse_args())

    models = Models(load_models=False)
    models._load_paddle_model(HardwareType.CONDA_GPU)
    injector = Injector(None, None, models, None)

    main(args["image"], injector)
