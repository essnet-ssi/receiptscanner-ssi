
from ocr_microservice.ocr_pipeline.rule_based_extractor.extractors.product_extractor import extract_products, clean_product_list
from ocr_microservice.ocr_pipeline.rule_based_extractor.extractors.date_extractor import extract_date
from ocr_microservice.ocr_pipeline.rule_based_extractor.extractors.shop_extractor import extract_shop_name
from ocr_microservice.ocr_pipeline.rule_based_extractor.extractors.total_price_extractor import extract_total_price


def process(image_results: list[dict], pre_processed_images: list, combined_result: dict) -> dict:

    ocr_strings = [item['ocr'][0] for item in image_results[0]['ocr_lilt']]
    ocr_lilt_list = image_results[0]['ocr_lilt']
    image_width, image_height = pre_processed_images[0].size

    products = extract_products(ocr_lilt_list, image_width, image_height)
    receipt_date = extract_date(ocr_strings)
    shop_name = extract_shop_name(ocr_strings)
    total_price = extract_total_price(ocr_lilt_list, products)
    products = clean_product_list(products, combined_result, total_price)
    result = {
        'products': products,
        'receipt_date': receipt_date,
        'shop_name': shop_name,
        'total_price': total_price
    }
    return result