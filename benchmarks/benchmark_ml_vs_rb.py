#%%
import os
from PIL import Image
from pathlib import Path
import time

from ocr_microservice.ocr_pipeline.pipeline_image import PipelineImage
from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.config.default import Config, Pipeline, Models, Cache
from ocr_microservice.ocr_pipeline.ocr_pipeline import process

image_dir = ''
output_dir = ''
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
injector = Injector(Config(save_results=True, data_dir=output_dir), Pipeline(), Models(), Cache())

for receipt_index, image_path in enumerate():
    
    print(receipt_index)
    start_time = time.time()
    
    image = Image.open(image_path)
    pipeline_image = PipelineImage(image, Path(image_path).stem)
    extracted_receipt, combined_result, image_results, pre_processed_images = process(pipeline_image, injector, True)
    

    # Check for bad result and redo without pre-processing with rotated image
    product_string = ''
    if extracted_receipt['products'] is not None:
        for product in extracted_receipt['products']:
            if 'product' in product and product['product'] is not None:
                for word in product['product']:
                    product_string += word
                break
    bad_result = product_string == '' and extracted_receipt['shop_name'] is None
    if bad_result:
        pipeline_image = PipelineImage(pre_processed_images[0].rotate(180), Path(image_path).stem)
        extracted_receipt, combined_result, image_results, pre_processed_images = process(pipeline_image, injector, False)

    display(pre_processed_images[0]) 

    # Extract ML-based results
    result = combined_result['receipt']
    date = result['date']['corrected'] if result['date'] and result['date']['corrected'] else result['date']['text'] if result['date'] else "N/A"
    store = result['store']['name']['corrected'] if result['store'] and result['store']['name'] and result['store']['name']['corrected'] else result['store']['name']['text'] if result['store'] and result['store']['name'] else "N/A"
    total_price = result['total']['price']['corrected'] if result['total'] and result['total']['price'] and result['total']['price']['corrected'] else result['total']['price']['text'] if result['total'] and result['total']['price'] else "N/A"
    if date is None:
        date = "N/A"
    if store is None:
        store = "N/A"
    if total_price is None:
        total_price = "N/A"
    
    articles = []
    if result['item_table']:
        for article in result['item_table']:
            article_name = article['description']['text'][0].replace('±', ' ') if article['description'] else "N/A"
            article_price = article['price']['corrected'] if article['price'] and article['price']['corrected'] else article['price']['text'] if article['price'] else "N/A"
            articles.append({'name': article_name, 'price': article_price})

    try:
        if isinstance(total_price, str):  # If it's a string, validate and convert
            total_price_float = float(total_price) if total_price.replace('.', '', 1).isdigit() else None
        elif isinstance(total_price, (int, float)):  # If it's already a number
            total_price_float = float(total_price)
        else:  # If it's neither, set it to None
            total_price_float = None
    except (ValueError, TypeError):
        total_price_float = None


    end_time = time.time()  # Record the end time
    duration = end_time - start_time

    print(image_path)
    print('')

    print('ML-based Results')
    print('')
    print(f"{'Date:':<{15}} {date:<{30}}")
    print(f"{'Store:':<{15}} {store.replace('±', ' '):<{30}}")
    if total_price_float is not None:
        print(f"{'Total Price:':<{15}} {total_price_float:<{30}.2f}")
    else:
        print(f"{'Total Price:':<{15}} {'N/A':<{30}}")
    print('')
    print(f"{'Product':<60} {'Price':>10}")
    print("=" * 75)
    for article in articles:
        article_name = article['name'] if article['name'] else "N/A"
        article_price = article['price'] if article['price'] else "N/A"
        print(f"{article_name:<60} {article_price:>10}")

    print('\n'*3)

    # Extract rule-based results
    receipt_date = extracted_receipt.get('receipt_date', 'N/A') or 'N/A'
    shop_name = extracted_receipt.get('shop_name', 'N/A') or 'N/A'
    total_price = extracted_receipt.get('total_price', 'N/A') or 'N/A'

    if total_price is float:
        total_price = round(total_price, 2)

    print('')
    print(f"{'Date:':<{15}} {receipt_date:<{30}}")
    print(f"{'Store:':<{15}} {str(shop_name):<{30}}")
    if total_price is not None:
        print(f"{'Total Price:':<{15}} {total_price_float:<{30}}")
    else:
        print(f"{'Total Price:':<{15}} {'N/A':<{30}}")
    print(f"{'Total Price:':<{15}} {total_price_float:<{30}}")
    print('')
    print(f"{'Product':<60} {'Price':>10}")
    print("=" * 75)
    if 'products' in extracted_receipt and extracted_receipt['products'] is not None:
        for product in extracted_receipt['products']:
            if len(product['product']) == 0:
                product['product'] = ['N/A']
            product_name = " ".join(product.get('product', 'N/A'))
            product_price = str(product.get('price', 'N/A'))
            print(f"{product_name:<60} {product_price:>10}")



    result_text = f"""
Process time: {duration:.2f} seconds

ML-based Results:

{'Date:':<15} {date:<30}
{'Store:':<15} {store:<30}
{'Total Price:':<15} {total_price:<30}

{"Product":<60} {"Price":>10}
{'=' * 75}
"""
    for article in articles:
        article_name = article['name'] if article['name'] else "N/A"
        article_price = article['price'] if article['price'] else "N/A"
        result_text += f"{article_name:<60} {article_price:>10}\n"

    result_text += f"""

Rule-based Results:

{'Date:':<15} {receipt_date:<30}
{'Store:':<15} {shop_name:<30}
{'Total Price:':<15} {total_price:<30}

{"Product":<60} {"Price":>10}
{'=' * 75}
"""
    if 'products' in extracted_receipt and extracted_receipt['products'] is not None:
        for product in extracted_receipt['products']:
            product_name = " ".join(product.get('product', ['N/A']))
            product_price = product.get('price', 'N/A')
            result_text += f"{product_name:<60} {product_price:>10}\n"

    txt_file_path = os.path.join(output_dir, f"{receipt_index}.txt")
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(result_text)

    # Save the preprocessed image(s)
    for idx, img in enumerate(pre_processed_images):
        img_path = os.path.join(output_dir, f"{receipt_index}.png")
        img.save(img_path)
