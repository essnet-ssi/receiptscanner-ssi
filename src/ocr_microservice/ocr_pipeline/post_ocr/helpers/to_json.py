from __future__ import annotations
from transformers import LiltForTokenClassification, LayoutLMv3Processor, LayoutLMv3ImageProcessor, AutoTokenizer
from pathlib import Path
import json
from typing import TYPE_CHECKING
from PIL import Image
from typing import List

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.post_ocr.helpers.label_groups import LabelGroups

class Receipt:
    data = None
    def __init__(self, labelGroups, data) -> None:
        self.data = data
        self.labelGroups = labelGroups
    
    def collect(self,labels):
        results = []
        for entry in self.data['ocr_lilt']:
            (_,label_str,lilt_confidence) = entry['lilt'][0]
            (text,bbox,ocr_confidence) = entry['ocr']
            if label_str in labels:
                results.append(entry)
        return results
    
    def make_text_dict(self,entries):
        text_map = {}
        for e in entries:
            (_,label_str,lilt_confidence) = e['lilt'][0]
            (text,bbox,ocr_confidence) = e['ocr']
            corrected = None
            if "corrected" in e:
                corrected = e['corrected']
            if text in text_map:
                e2 = text_map[text]
                ocr_c = round(e2['ocr_confidence'] + (1-e2['ocr_confidence'])*ocr_confidence,3)
                lilt_c = round(e2['lilt_confidence'] + (1-e2['lilt_confidence'])*lilt_confidence,3)
                text_map[text] = {"text": text, "corrected": corrected, "bbox":e2['bbox'] + [bbox], "ocr_confidence": ocr_c, "lilt_confidence": lilt_c}
            else:
                text_map[text] = {"text": text, "corrected": corrected, "bbox":[bbox], "ocr_confidence": ocr_confidence, "lilt_confidence": lilt_confidence}
        
        return text_map,entries
    
    def select_best(self,input):
        text_map,metadata = input

        if not len(text_map):
            return None

        # sort on lilt confidence
        text_map = {k: v for k, v in sorted(text_map.items(), key=lambda item: item[1]['lilt_confidence'], reverse=True)}

        # select the best entry
        best = next(iter(text_map.values()))

        result = {}
        result['text'] = best['text']
        result['corrected'] = best['corrected']
        result['bbox'] = best['bbox']
        result['ocr_confidence'] = best['ocr_confidence']
        result['du_confidence'] = best['lilt_confidence']
        result['metadata'] = metadata

        return result
    
    def select_all(self,input):
        text_map,metadata = input

        if not len(text_map):
            return None
        
        result = {}
        result['text'] = []
        result['bbox'] = []
        result['ocr_confidence'] = 1.0
        result['du_confidence'] = 1.0
        result['metadata'] = metadata
        for text,v in text_map.items():
            result['text'].append(text)
            result['bbox'].append(v["bbox"])
            result['ocr_confidence'] = round(result['ocr_confidence']*v["ocr_confidence"],3)
            result['du_confidence'] = round(result['du_confidence']*v["lilt_confidence"],3)

        return result

    def collect_lines(self,group):
        lines = []
        for l in self.data['lines']:
            line_info = l['line_info']
            conflict = line_info['line_conflict']
            merged_labels = line_info['merged_labels']
            line_group,confidence = merged_labels[0]
            if line_group == group:
                lines.append(l)
        return lines
    
    # tries to find a label in the lilt labels
    # it ignores labels belonging to other groups
    # returns a list of tuples (ocr_lilt entry, lilt_entry)
    def collect_in_table_line(self,line,label):
        entries = line['ocr_lilt']
        result = []
        for e in entries:
            lilt = e['lilt']
            for l in lilt:
                _,label_str,_ = l
                if label == label_str:
                    result.append((e,l))
                elif self.labelGroups.doBelongToSameGroup([label_str,label]):
                    break # same group but different label -> no hit -> break out of inner loop: go to next entry
        return result
    
    def select_all_in_table_line(self,entries):
        if not entries:
            return None

        result = {}
        result['text'] = []
        result['bbox'] = []
        result['ocr_confidence'] = 1.0
        result['du_confidence'] = 1.0
        for entry,lilt_entry in entries:
            text,bbox,ocr_confidence = entry['ocr']
            label_id,label_str,lilt_confidence = lilt_entry
            result['text'].append(text)
            result['bbox'].append(bbox)
            result['ocr_confidence'] = round(result['ocr_confidence'] * ocr_confidence,3)
            result['du_confidence'] = round(result['du_confidence'] * lilt_confidence,3)
        
        return result
    
    def select_best_in_table_line(self,entries):
        if not entries:
            return None

        best_entry = None
        best_lilt_entry = None
        best_lilt_confidence = 0.0
        for entry,lilt_entry in entries:
            _,_,lilt_confidence = lilt_entry
            if not best_entry or (best_entry and lilt_confidence > best_lilt_confidence):
                best_entry = entry
                best_lilt_entry = lilt_entry
                best_lilt_confidence = lilt_confidence
        
        text,bbox,ocr_confidence = best_entry['ocr']
        label_id,label_str,lilt_confidence = best_lilt_entry
        result = {}
        result['text'] = text
        result['corrected'] = best_entry['corrected']
        result['bbox'] = bbox
        result['ocr_confidence'] = ocr_confidence
        result['du_confidence'] = lilt_confidence
        return result
    
    def date(self):
        return self.select_best(self.make_text_dict(self.collect(["I-date_value"])))

    def time(self):
        return self.select_best(self.make_text_dict(self.collect(["I-time_value"])))
    
    def parse_tax_table_line(self,line):
        result = {}
        result['description'] = self.select_all_in_table_line(self.collect_in_table_line(line,"I-tax.description"))
        result['price'] = self.select_best_in_table_line(self.collect_in_table_line(line,"I-tax.price"))
        result['price_excl'] = self.select_best_in_table_line(self.collect_in_table_line(line,"I-tax.price_excl"))
        result['price_incl'] = self.select_best_in_table_line(self.collect_in_table_line(line,"I-tax.price_incl"))
        result["metadata"] = line
        return result

    def tax_table(self):
        result = []
        lines = self.collect_lines("tax_table")
        for l in lines:
            result.append(self.parse_tax_table_line(l))
        return result

    def store(self):
        result = {}
        result['name'] = self.select_best(self.make_text_dict(self.collect(["I-store.name"])))
        result['address'] = self.select_all(self.make_text_dict(self.collect(["I-store.address"])))
        result['phone'] = self.select_best(self.make_text_dict(self.collect(["I-store.phone"])))
        result['email'] = self.select_best(self.make_text_dict(self.collect(["I-store.email"])))
        result['website'] = self.select_best(self.make_text_dict(self.collect(["I-store.website"])))
        result['tax_id'] = self.select_best(self.make_text_dict(self.collect(["I-store.tax_id"])))
        result['etc'] = self.select_all(self.make_text_dict(self.collect(["I-store.etc"])))
        return result
    
    def parse_item_table_line(self,line):
        result = {}
        result['quantity'] = self.select_best_in_table_line(self.collect_in_table_line(line,"I-item.quantity"))
        result['description'] = self.select_all_in_table_line(self.collect_in_table_line(line,"I-item.description"))
        result['unit_price'] = self.select_best_in_table_line(self.collect_in_table_line(line,"I-item.unit_price"))
        result['price'] = self.select_best_in_table_line(self.collect_in_table_line(line,"I-item.price"))
        result['id'] = self.select_best_in_table_line(self.collect_in_table_line(line,"I-item.id"))
        result['discount_description'] = self.select_all_in_table_line(self.collect_in_table_line(line,"I-item.discount_description"))
        result['discount_price'] = self.select_best_in_table_line(self.collect_in_table_line(line,"I-item.discount_price"))
        result['etc'] = self.select_best_in_table_line(self.collect_in_table_line(line,"I-item.etc"))
        result["metadata"] = line
        return result
    
    def item_table(self):
        result = []
        lines = self.collect_lines("item_table")
        for l in lines:
            result.append(self.parse_item_table_line(l))
        return result
    
    def sub_total(self):
        result = {}
        result['price'] = self.select_best(self.make_text_dict(self.collect(["I-sub_total.price"])))
        result['discount_price'] = self.select_best(self.make_text_dict(self.collect(["I-sub_total.discount_price"])))
        result['discount_item_text'] = self.select_all(self.make_text_dict(self.collect(["I-sub_total.discount_item_text"])))
        result['discount_item_price'] = self.select_all(self.make_text_dict(self.collect(["I-sub_total.discount_item_price"])))
        result['tax_price'] = self.select_best(self.make_text_dict(self.collect(["I-sub_total.tax_price"])))
        result['tax_excl_price'] = self.select_best(self.make_text_dict(self.collect(["I-sub_total.tax_excl_price"])))
        result['tax_incl_price'] = self.select_best(self.make_text_dict(self.collect(["I-sub_total.tax_incl_price"])))
        result['service_price'] = self.select_best(self.make_text_dict(self.collect(["I-sub_total.service_price"])))
        result['item_rows'] = self.select_best(self.make_text_dict(self.collect(["I-sub_total.item_rows_value"])))
        result['quantity'] = self.select_best(self.make_text_dict(self.collect(["I-sub_total.quantity_value"])))
        result['etc_price'] = self.select_best(self.make_text_dict(self.collect(["I-sub_total.etc_price"])))
        return result

    def total(self):
        result = {}
        result['price'] = self.select_best(self.make_text_dict(self.collect(["I-total.price","I-payment.other_price","I-payment.details_total_price"])))
        result['rounded_price'] = self.select_best(self.make_text_dict(self.collect(["I-total.rounded_price"])))
        result['etc_price'] = self.select_best(self.make_text_dict(self.collect(["I-total.etc_price"])))
        return result
    
    def payment(self):
        result = {}
        result['cash_price'] = self.select_best(self.make_text_dict(self.collect(["I-payment.cash_price"])))
        result['change_price'] = self.select_best(self.make_text_dict(self.collect(["I-payment.change_price"])))
        result['other_price'] = self.select_best(self.make_text_dict(self.collect(["I-payment.other_price"])))
        result['details_total_price'] = self.select_best(self.make_text_dict(self.collect(["I-payment.details_total_price"])))
        result['etc_price'] = self.select_best(self.make_text_dict(self.collect(["I-payment.etc_price"])))
        return result
    
    def get(self):
        output = {}

        # print("Parsing date")
        output['date'] = self.date()
        # print("Parsing time")
        output['time'] = self.time()
        # print("Parsing tax table")
        output['tax_table'] = self.tax_table()
        # print("Parsing store")
        output['store'] = self.store()
        # print("Parsing item table")
        output['item_table'] = self.item_table()
        # print("Parsing sub_total")
        output['sub_total'] = self.sub_total()
        # print("Parsing total")
        output['total'] = self.total()
        # print("Parsing payment")
        output['payment'] = self.payment()

        return output

def to_json(images: List[Image.Image], filename: str, image_results: list, injector: Injector):
    # combine data
    input_data = {
        "ocr_lilt": [],
        "lines": []
    }
    for index, _ in enumerate(images):
        d = image_results[index]
        for entry in d['ocr_lilt']:
            entry['image'] = f'{filename}_{index}'
            input_data['ocr_lilt'].append(entry)
        for line in d['lines']:
            line['image'] = f'{filename}_{index}'
            input_data['lines'].append(line)

    model = injector.models.lilt_model
    labelGroups = LabelGroups(model.config.id2label, model.config.label2id)
    receipt = Receipt(labelGroups, input_data)

    output = {}
    #output['metadata'] = image_results
    output['receipt'] = receipt.get()

    return output
