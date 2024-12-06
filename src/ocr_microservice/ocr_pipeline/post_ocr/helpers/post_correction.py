from __future__ import annotations
import re
import dateutil.parser as parser
from enum import Enum
from PIL import Image
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector


class LabelFormat(Enum):
    TEXT = 1
    DATEFORMAT = 2
    TIMEFORMAT = 3
    DATETIMEFORMAT = 4
    PRICE = 5
    NUMBER = 6
    MIXED = 7
    LOCATION = 8

labelTypeMap = {
    'O': LabelFormat.MIXED,
    'I-date_text':LabelFormat.TEXT,
    'I-date_value':LabelFormat.DATEFORMAT,
    'I-time_text':LabelFormat.TEXT,
    'I-time_value':LabelFormat.TIMEFORMAT,
    'I-datetime':LabelFormat.DATETIMEFORMAT,
    'I-heading':LabelFormat.MIXED,
    'I-tax.header':LabelFormat.TEXT,
    'I-tax.description':LabelFormat.MIXED,
    'I-tax.price':LabelFormat.PRICE,
    'I-tax.price_excl':LabelFormat.PRICE,
    'I-tax.price_incl':LabelFormat.PRICE,
    'I-store.name':LabelFormat.MIXED,
    'I-store.address':LabelFormat.LOCATION,
    'I-store.phone':LabelFormat.MIXED,
    'I-store.email':LabelFormat.MIXED,
    'I-store.website':LabelFormat.MIXED,
    'I-store.tax_id':LabelFormat.MIXED,
    'I-store.etc':LabelFormat.MIXED,
    'I-item.header':LabelFormat.TEXT,
    'I-item.quantity':LabelFormat.NUMBER,
    'I-item.description':LabelFormat.MIXED,
    'I-item.unit_price':LabelFormat.PRICE,
    'I-item.price':LabelFormat.PRICE,
    'I-item.id':LabelFormat.MIXED,
    'I-item.discount_description':LabelFormat.MIXED,
    'I-item.discount_price':LabelFormat.PRICE,
    'I-item.etc':LabelFormat.MIXED,
    'I-sub_total.text':LabelFormat.TEXT,
    'I-sub_total.price':LabelFormat.PRICE,
    'I-sub_total.discount_text':LabelFormat.TEXT,
    'I-sub_total.discount_price':LabelFormat.PRICE,
    'I-sub_total.discount_item_text':LabelFormat.MIXED,
    'I-sub_total.discount_item_price':LabelFormat.PRICE,
    'I-sub_total.tax_text':LabelFormat.TEXT,
    'I-sub_total.tax_price':LabelFormat.PRICE,
    'I-sub_total.tax_excl_text':LabelFormat.TEXT,
    'I-sub_total.tax_excl_price':LabelFormat.PRICE,
    'I-sub_total.tax_incl_text':LabelFormat.TEXT,
    'I-sub_total.tax_incl_price':LabelFormat.PRICE,
    'I-sub_total.service_text':LabelFormat.MIXED,
    'I-sub_total.service_price':LabelFormat.PRICE,
    'I-sub_total.item_rows_text':LabelFormat.MIXED,
    'I-sub_total.item_rows_value':LabelFormat.NUMBER,
    'I-sub_total.quantity_text':LabelFormat.TEXT,
    'I-sub_total.quantity_value':LabelFormat.NUMBER,
    'I-sub_total.etc_text':LabelFormat.MIXED,
    'I-sub_total.etc_price':LabelFormat.PRICE,
    'I-total.text':LabelFormat.TEXT,
    'I-total.price':LabelFormat.PRICE,
    'I-total.rounded_text':LabelFormat.TEXT,
    'I-total.rounded_price':LabelFormat.PRICE,
    'I-payment.cash_text':LabelFormat.TEXT,
    'I-payment.cash_price':LabelFormat.PRICE,
    'I-payment.change_text':LabelFormat.TEXT,
    'I-payment.change_price':LabelFormat.PRICE,
    'I-payment.other_text':LabelFormat.TEXT,
    'I-payment.other_price':LabelFormat.PRICE,
    'I-payment.details_total_text':LabelFormat.TEXT,
    'I-payment.details_total_price':LabelFormat.PRICE,
    'I-payment.etc_text':LabelFormat.TEXT,
    'I-payment.etc_price':LabelFormat.PRICE
}

def getLabelFromDict(stringtype):
    return labelTypeMap[stringtype]

def priceTypeFormatter(inputstring):
    replaced = inputstring.replace("l", "1")
    replaced = replaced.replace("I", "1")
    replaced = replaced.replace("o", "0")
    replaced = replaced.replace("O", "0")
    replaced = replaced.replace("S", "5")
    replaced = replaced.replace("b", "6")
    replaced = replaced.replace("G", "6") 
    replaced = replaced.replace("B", "8")
    return replaced

def datetimeRegex(input_string):
    format="%Y-%m-%d"
    try:
        return parser.parse(input_string).strftime(format)
    except:
        date_regex = re.search("\d{2}(\.|\-|\/)\d{2}(\.|\-|\/)\d{2,4}", input_string)
        if date_regex:
            date_regex = input_string[date_regex.start():date_regex.end()]
            try:
                return parser.parse(str(date_regex)).strftime(format)
            except:
                return None

def to_float(price):
    price = price.replace(',','.')

    sep = price.rfind('.')
    if sep == -1:
        res_regex = re.search("\d+", price)
        if res_regex:
            return float(price[res_regex.start():res_regex.end()])
        else:
            return None

    first_part = price[:sep]
    second_part = price[sep+1:]

    # first part
    res_regex = re.search("-?\d+$", first_part)
    if res_regex:
        first_part = first_part[res_regex.start():res_regex.end()]

    # second part
    res_regex = re.search("^\d+", second_part)
    if res_regex:
        second_part = second_part[res_regex.start():res_regex.end()]

    try:
        return float(first_part + '.' + second_part)
    except:
        return 0.0

def to_int(str_number):
    res_regex = re.search("\d+", str_number)
    if res_regex:
        return int(str_number[res_regex.start():res_regex.end()])
    else:
        return None

def post_correction(image: Image.Image, data: dict, injector: Injector) -> Tuple[Image.Image, dict]:
    for item in data["ocr_lilt"]:
        item['corrected'] = None

        stringtype = str(item['lilt'][0][1])
        stringvalue = str(item['ocr'][0])

        match getLabelFromDict(stringtype):
            case LabelFormat.DATEFORMAT:
                formatted_date = datetimeRegex(stringvalue)
                if formatted_date != None:
                    item['corrected'] = formatted_date
            case LabelFormat.DATETIMEFORMAT:
                formatted_date = datetimeRegex(stringvalue)
                if formatted_date != None:
                    item['corrected'] = formatted_date
            case LabelFormat.PRICE:
                price = priceTypeFormatter(stringvalue)
                item['corrected'] = to_float(price)
            case LabelFormat.NUMBER:
                number = priceTypeFormatter(stringvalue)
                item['corrected'] = to_int(number)
                    
    return image, data
