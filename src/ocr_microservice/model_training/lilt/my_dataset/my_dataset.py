'''
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
'''
import os
import csv
from PIL import Image
import datasets

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

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

def json_lookup(json_data, image_path):
    image_filename = os.path.basename(image_path)
    # [9:] removes the trailing hash from the filename
    result_list = list(filter(lambda x:x["file_upload"][9:]==image_filename,json_data))
    if not len(result_list):
        print(image_path)
    assert(len(result_list))
    return result_list[0]

def json_tokens(entry):
    annotations = entry['annotations'][0]['result']
    result = []
    for a in annotations:
        if a["type"] == "textarea":
            result.append(a["value"]["text"][0])
    return result

def json_bboxes(entry):
    annotations = entry['annotations'][0]['result']
    result = []
    for a in annotations:
        if a["type"] == "rectangle":
            result.append(normalize_bbox(a["value"],a["original_width"],a["original_height"]))
    return result

label2ner_tag = {
    "O": 0,
    "I-date_text": 1,
    "I-date_value": 2,
    "I-time_text": 3,
    "I-time_value": 4,
    "I-datetime": 5,
    "I-heading": 6,
    "I-unused8": 7,
    "I-unused7": 8,
    "I-unused6": 9,
    "I-tax.header": 10,
    "I-tax.description": 11,
    "I-tax.price": 12,
    "I-tax.price_excl": 13,
    "I-tax.price_incl": 14,
    "I-unused5": 15,
    "I-unused4": 16,
    "I-unused3": 17,
    "I-unused2": 18,
    "I-unused1": 19,
    "I-store.name": 20,
    "I-store.address": 21,
    "I-store.phone": 22,
    "I-store.email": 23,
    "I-store.website": 24,
    "I-store.tax_id": 25,
    "I-store.unused3": 26,
    "I-store.unused2": 27,
    "I-store.unused1": 28,
    "I-store.etc": 29,
    "I-item.header": 30,
    "I-item.quantity": 31,
    "I-item.description": 32,
    "I-item.unit_price": 33,
    "I-item.price": 34,
    "I-item.id": 35,
    "I-item.discount_description": 36,
    "I-item.discount_price": 37,
    "I-item.etc": 38,
    "I-item.unused11": 39,
    "I-item.unused10": 40,
    "I-item.unused9": 41,
    "I-item.unused8": 42,
    "I-item.unused7": 43,
    "I-item.unused6": 44,
    "I-item.unused5": 45,
    "I-item.unused4": 46,
    "I-item.unused3": 47,
    "I-item.unused2": 48,
    "I-item.unused1": 49,
    "I-sub_total.text": 50,
    "I-sub_total.price": 51,
    "I-sub_total.discount_text": 52,
    "I-sub_total.discount_price": 53,
    "I-sub_total.discount_item_text": 54,
    "I-sub_total.discount_item_price": 55,
    "I-sub_total.tax_text": 56,
    "I-sub_total.tax_price": 57,
    "I-sub_total.tax_excl_text": 58,
    "I-sub_total.tax_excl_price": 59,
    "I-sub_total.tax_incl_text": 60,
    "I-sub_total.tax_incl_price": 61,
    "I-sub_total.service_text": 62,
    "I-sub_total.service_price": 63,
    "I-sub_total.item_rows_text": 64,
    "I-sub_total.item_rows_value": 65,
    "I-sub_total.quantity_text": 66,
    "I-sub_total.quantity_value": 67,
    "I-sub_total.etc_text": 68,
    "I-sub_total.etc_price": 69,
    "I-total.text": 70,
    "I-total.price": 71,
    "I-total.rounded_text": 72,
    "I-total.rounded_price": 73,
    "I-total.unused4": 74,
    "I-total.unused3": 75,
    "I-total.unused2": 76,
    "I-total.unused1": 77,
    "I-total.etc_text": 78,
    "I-total.etc_price": 79,
    "I-payment.cash_text": 80,
    "I-payment.cash_price": 81,
    "I-payment.change_text": 82,
    "I-payment.change_price": 83,
    "I-payment.other_text": 84,
    "I-payment.other_price": 85,
    "I-payment.details_total_text": 86,
    "I-payment.details_total_price": 87,
    "I-payment.etc_text": 88,
    "I-payment.etc_price": 89
}

def json_ner_tags(entry):
    annotations = entry['annotations'][0]['result']
    result = []
    for a in annotations:
        if a["type"] == "labels":
            result.append(label2ner_tag[a["value"]["labels"][0]])
    return result

logger = datasets.logging.get_logger(__name__)


class BuilderConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(BuilderConfig, self).__init__(**kwargs)


class MyDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(name="my_dataset", version=datasets.Version("1.0.0"), description="LiLT dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="MyDataset",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "I-date_text",
                                "I-date_value",
                                "I-time_text",
                                "I-time_value",
                                "I-datetime",
                                "I-heading",
                                "I-unused8",
                                "I-unused7",
                                "I-unused6",
                                "I-tax.header",
                                "I-tax.description",
                                "I-tax.price",
                                "I-tax.price_excl",
                                "I-tax.price_incl",
                                "I-unused5",
                                "I-unused4",
                                "I-unused3",
                                "I-unused2",
                                "I-unused1",
                                "I-store.name",
                                "I-store.address",
                                "I-store.phone",
                                "I-store.email",
                                "I-store.website",
                                "I-store.tax_id",
                                "I-store.unused3",
                                "I-store.unused2",
                                "I-store.unused1",
                                "I-store.etc",
                                "I-item.header",
                                "I-item.quantity",
                                "I-item.description",
                                "I-item.unit_price",
                                "I-item.price",
                                "I-item.id",
                                "I-item.discount_description",
                                "I-item.discount_price",
                                "I-item.etc",
                                "I-item.unused11",
                                "I-item.unused10",
                                "I-item.unused9",
                                "I-item.unused8",
                                "I-item.unused7",
                                "I-item.unused6",
                                "I-item.unused5",
                                "I-item.unused4",
                                "I-item.unused3",
                                "I-item.unused2",
                                "I-item.unused1",
                                "I-sub_total.text",
                                "I-sub_total.price",
                                "I-sub_total.discount_text",
                                "I-sub_total.discount_price",
                                "I-sub_total.discount_item_text",
                                "I-sub_total.discount_item_price",
                                "I-sub_total.tax_text",
                                "I-sub_total.tax_price",
                                "I-sub_total.tax_excl_text",
                                "I-sub_total.tax_excl_price",
                                "I-sub_total.tax_incl_text",
                                "I-sub_total.tax_incl_price",
                                "I-sub_total.service_text",
                                "I-sub_total.service_price",
                                "I-sub_total.item_rows_text",
                                "I-sub_total.item_rows_value",
                                "I-sub_total.quantity_text",
                                "I-sub_total.quantity_value",
                                "I-sub_total.etc_text",
                                "I-sub_total.etc_price",
                                "I-total.text",
                                "I-total.price",
                                "I-total.rounded_text",
                                "I-total.rounded_price",
                                "I-total.unused4",
                                "I-total.unused3",
                                "I-total.unused2",
                                "I-total.unused1",
                                "I-total.etc_text",
                                "I-total.etc_price",
                                "I-payment.cash_text",
                                "I-payment.cash_price",
                                "I-payment.change_text",
                                "I-payment.change_price",
                                "I-payment.other_text",
                                "I-payment.other_price",
                                "I-payment.details_total_text",
                                "I-payment.details_total_price",
                                "I-payment.etc_text",
                                "I-payment.etc_price"
                            ]
                        )
                    ),
                    "image_path": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"my_dataset/train/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"my_dataset/test/"}
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def parse_csv(self,csv_in):
        data = {}
        f = open(csv_in,'r')
        reader = csv.reader(f)

        id = 0
        img_path = ''
        words = []
        bboxes = []
        ner_tags = []
        for row in reader:
            cur_img_path = row[0]
            if img_path == '':
                img_path = cur_img_path
            im = Image.open(cur_img_path)
            width, height = im.size
            bbox = normalize_bbox(eval(row[1]),width,height)

            text,confidence = eval(row[2])
            ner = eval(row[3])

            if cur_img_path != img_path:
                data[os.path.basename(img_path)] = {
                    "id": id,
                    "words": words,
                    "bboxes": bboxes,
                    "ner_tags": ner_tags
                }
                words = []
                bboxes = []
                ner_tags = []
                id += 1
        
            words.append(text)
            bboxes.append(bbox)
            ner_tags.append(ner)

            img_path = cur_img_path

        if img_path != '':
                data[os.path.basename(img_path)] = {
                    "id": id,
                    "words": words,
                    "bboxes": bboxes,
                    "ner_tags": ner_tags
                }
        
        return data

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)

        data = self.parse_csv("my_dataset/data.csv")

        for file in sorted(os.listdir(filepath)):
            image_path = os.path.join(filepath, file)
            image, size = load_image(image_path)
            entry = data[file]
            id = entry["id"]
            words = entry["words"]
            bboxes = entry["bboxes"]
            ner_tags = entry["ner_tags"]
            yield id,{"id": str(id), "words": words, "bboxes": bboxes, "ner_tags": ner_tags, "image_path": image_path}
