import os
from PIL import Image
from helpers import clear_previous_results, get_image_paths
from model.basic_receipt import Receipt, Article

from pathlib import Path
import argparse
import json
import time
import numpy as np

from ocr_microservice.ocr_pipeline.pre_ocr import pre_ocr
from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.config.default import Config, Pipeline, Models, Cache
from ocr_microservice.ocr_pipeline.ocr_pipeline import process

MIN_ACCURACY = 0.7


def get_data_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_dir = os.path.join(parent_dir, "benchmarks/assets")
    return data_dir

def get_image(datadir: str, filename: str = "test_image_1.jpg"):
    image_path = os.path.join(data_dir, filename)
    image = Image.open(image_path)
    return image

def compare_lilt_label(item, lilt_label: str):
    return item['lilt'][0][1] == lilt_label and item['lilt'][0][2] >= MIN_ACCURACY

if __name__ == "__main__":
    data_dir = get_data_dir()
    eval_json_path = os.path.join(data_dir, "annotation.json")
    f = open(eval_json_path)
    eval_json = json.load(f) # Opening the annotated JSON
    image_paths = []
    calculation_times = []

    for item in eval_json:
        image_paths.append(os.path.join(data_dir, item))
    
    injector = Injector(Config(), Pipeline(), Models(), Cache())

    receipts = {}

    for path in image_paths: 
        image = Image.open(path)
        filename = Path(path).stem
        start_time = time.time()
        result = process(image, filename, injector)
        end_time = time.time()
        elapsed_time = end_time - start_time
        calculation_times.append(elapsed_time)

        vendor = ''
        address = ''
        date = ''
        articles = []
        total = ''

        for item in result['ocr_lilt']:
            if compare_lilt_label(item,'I-store.name'):
                vendor = item['ocr'][0]
            if compare_lilt_label(item,'I-store.address'):
                address = address + ' ' + item['ocr'][0]
            if compare_lilt_label(item,'I-datetime'):
                date = item['ocr'][0]
            if compare_lilt_label(item,'I-date_value') and date == '':
                date = item['ocr'][0]
            if compare_lilt_label(item,'I-total.price'):
                total = float(item['corrected'])

        for item in result['lines']:
            if item['line_info']['merged_labels'][0][0] == 'item_table':
                description = ''
                quantity = None
                tot_art_price = None
                unit_price = None
                for ocr_label in item['ocr_lilt']:
                    if compare_lilt_label(ocr_label,'I-item.price'):
                        total_price = ocr_label['ocr'][0]
                    if compare_lilt_label(ocr_label, 'I-item.unit_price'):
                        tot_art_price = ocr_label['ocr'][0]
                    if compare_lilt_label(ocr_label,'I-item.description'):
                        description = description + ' ' + ocr_label['ocr'][0]
                    if compare_lilt_label(ocr_label,'I-item.quantity'):
                        quantity = ocr_label['ocr'][0]

                isValidArticle = sum(x is not None for x in [description, quantity, tot_art_price, unit_price])
                if isValidArticle >= 2:
                    articles.append(Article(description = description, quantity= quantity, tot_art_price= tot_art_price, unit_price= unit_price))

        receipt = Receipt(vendor = vendor, address = address, date = date.replace("/", "/"), articles = articles, total = total)
        receipts[filename] = receipt


    total_comparison_counter = 0
    for receipt in receipts:
        comparison_counter = 0
        eval_receipt = eval_json[receipt + '.JPG']
        print(receipt)
        print('Date comparison, real: ' + eval_receipt['date'] + ', found: '+ str(receipts[receipt].date))
        print('Artikel count, real: ' + str(len(eval_receipt['articles'])) + ', found: '+ str(len(receipts[receipt].articles)))
        print('Totaal comparison, real: ' + str(eval_receipt['total']) + ', found: '+ str(receipts[receipt].total))
        if eval_receipt['date'] == str(receipts[receipt].date):
            comparison_counter = comparison_counter+1
            total_comparison_counter = total_comparison_counter+1
        if len(eval_receipt['articles']) == len(receipts[receipt].articles):
            comparison_counter = comparison_counter+1
            total_comparison_counter = total_comparison_counter+1
        if eval_receipt['total'] == receipts[receipt].total:
            comparison_counter = comparison_counter+1
            total_comparison_counter = total_comparison_counter+1
        print('Correct: ' + str((comparison_counter/3*100))+ '%')
    
    print('Total accuracy: '  + str((total_comparison_counter/30*100))+ '%')   
    print('Total computing time: ' + str(np.sum(np.array(calculation_times))) + ', average computing time: ' + str(np.mean(np.array(calculation_times))))

