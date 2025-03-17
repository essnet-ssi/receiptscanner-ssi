import re
import statistics
import Levenshtein

special_store_label = {
        "Total price": ["Totaal", "Te Betalen", "Totaalprijs", "Total", "art.", "PIN", "TOTAAL", "VPAY", "Betaalwijze", "UPay"],
        "Other": ["Bankpas", "Pinnen", "Uw", "Voordeel", "Persoonlijke Bonus", "Subtotaal", "Bonus Box", "Maestro", "Betaalautomaat", "BONUSBOX"]
    }

remove_labels = ['EUR']

def filter_right_side(data, x_threshold):
    return [item for item in data if item['ocr'][1][0] > x_threshold]

def is_price(text):
    normalized = text.upper()
    normalized = normalized.split(' B')[0]
    normalized = normalized.split('B')[0]
    normalized = normalized.replace('O', '0').replace('I', '1').replace('L', '1')
    normalized = normalized.replace('S', '5').replace('B', '8').replace('G', '9')
    if re.search(r'[-/:]', normalized):
        return False
    price_regex = re.compile(r'(\d{1,3}[.,]\d{2})')
    match = price_regex.search(normalized)
    if match:
        return True
    if normalized.isdigit():
        return True
    return False


def filter_prices(data):
    filtered = []
    for item in data:
        text = item['ocr'][0] 
        if is_price(text):
            filtered.append(item)
    return filtered


def group_prices(filtered_prices, initial_threshold_ratio=0.4, height_threshold_ratio=0.4):
    """  
    :param filtered_prices: List of dictionaries with price information.
    :param initial_threshold_ratio: Initial ratio used to calculate vertical and x thresholds based on bounding box size.
    :param height_threshold_ratio: Ratio used to determine if heights are similar.
    """
    filtered_prices = sorted(filtered_prices, key=lambda x: x['ocr'][1][1])
    groups = []
    vertical_thresholds = []
    x_thresholds = []

    for entry in filtered_prices:
        value, bounding_box, confidence = entry['ocr']
        x1, y1, x2, y2 = bounding_box
        height = y2 - y1
        width = x2 - x1
        center_x = (x1 + x2) / 2
        added_to_group = False

        num_thresholds = len(vertical_thresholds)
        if num_thresholds == 0:
            vertical_threshold = initial_threshold_ratio * height
            x_threshold = initial_threshold_ratio * width
        else:
            mean_vertical = statistics.mean(vertical_thresholds)
            mean_x = statistics.mean(x_thresholds)
            std_vertical = statistics.stdev(vertical_thresholds) if num_thresholds > 1 else 0
            std_x = statistics.stdev(x_thresholds) if num_thresholds > 1 else 0
            weight = min(1, num_thresholds / 10)
            vertical_threshold = mean_vertical + weight * std_vertical
            x_threshold = mean_x + weight * std_x

        for group in groups:
            group_centers_x = [(item['ocr'][1][0] + item['ocr'][1][2]) / 2 for item in group]
            group_y_positions = [item['ocr'][1][1] for item in group]
            group_heights = [item['ocr'][1][3] - item['ocr'][1][1] for item in group]

            avg_group_center_x = statistics.mean(group_centers_x)
            avg_group_y = statistics.mean(group_y_positions)
            avg_group_height = statistics.mean(group_heights)

            vertical_proximity = abs(y1 - (group[-1]['ocr'][1][3])) <= vertical_threshold or abs(y1 - (group[-1]['ocr'][1][3])) <= 3 * avg_group_height
            height_similarity = abs(height - avg_group_height) <= height_threshold_ratio * avg_group_height
            x_position_similarity = abs(center_x - avg_group_center_x) <= 0.5 * x_threshold

            if vertical_proximity and height_similarity and x_position_similarity:
                group.append(entry)
                added_to_group = True
                break

        if not added_to_group:
            groups.append([entry])

        vertical_thresholds.append(height * initial_threshold_ratio)
        x_thresholds.append(width * initial_threshold_ratio)
    return merge_groups(groups)


def merge_groups(groups):
    merged_groups = []
    i = 0
    while i < len(groups):
        current_group = groups[i]
        while i + 1 < len(groups):
            next_group = groups[i + 1]
            _, current_bounding_box, _ = current_group[-1]['ocr']
            _, next_bounding_box, _ = next_group[0]['ocr']
            current_x1, current_y1, current_x2, current_y2 = current_bounding_box
            next_x1, next_y1, next_x2, next_y2 = next_bounding_box
            avg_group_height = statistics.mean([item['ocr'][1][3] - item['ocr'][1][1] for item in current_group])
            next_group_height = next_y2 - next_y1
            height_similarity = abs(next_group_height - avg_group_height) <= 0.25 * avg_group_height
            if (abs(next_y1 - current_y2) <= avg_group_height * 1.5) and abs(next_x1 - current_x1) <= 0.2 * avg_group_height and height_similarity:
                current_group.extend(next_group)
                i += 1
            else:
                break
        merged_groups.append(current_group)
        i += 1
    return merged_groups


def remove_duplicates(grouped_prices):
    seen_entries = set()
    unique_grouped_prices = []
    for group in grouped_prices:
        unique_group = []
        for entry in group:
            bounding_box = tuple(entry['ocr'][1])
            if bounding_box not in seen_entries:
                seen_entries.add(bounding_box)
                unique_group.append(entry)
        if unique_group:  
            unique_grouped_prices.append(unique_group)

    return unique_grouped_prices


def count_unit_price_labels(group):
    count = 0
    for entry in group:
        lilt_info = entry.get('lilt', [])
        count += sum(1 for _, label, _ in lilt_info if label in ['I-item.unit_price', 'I-item.price'])
    return count


def correct_price(price):
    price = price.replace(',', '.')
    price = price.replace('±', '')
    price = price.lower().split('b')[0]
    if price[-1].isdigit() == False:
        price = price[:-1]
    if '.' not in price and ',' not in price:
        if price.isdigit():
            if len(price) >= 3:
                price = str(float(price) / 100)
    parts = price.split('.')
    if len(parts) > 1:
        if len(parts) == 2 and len(parts[1]) > 2:
            price = f'{parts[0]}.{parts[1][0:2]}'
        elif len(parts[1]) == 1:
            price = price + '0'
    price = re.sub(r"[^\d.]", "", price)
    return price


def filter_by_middle_and_right(items, lower_bound, upper_bound, max_right):
    filtered_items = []
    for item in items:
        if 'ocr' in item:
            _, bbox, _ = item['ocr']
            if bbox and len(bbox) >= 4:
                middle = (bbox[1] + bbox[3]) / 2
                right = bbox[2]
                if lower_bound <= middle <= upper_bound and right < max_right:
                    filtered_items.append(item)
    return filtered_items


def preprocess_string(s):
    return s.lower().replace('±', ' ')


def count_letters(s):
    return sum(1 for char in s if char.isalpha())

def replace_1_with_case_based_l(input_string):
    letters = [char for char in input_string if char.isalpha()]
    uppercase_count = sum(1 for char in letters if char.isupper())
    lowercase_count = len(letters) - uppercase_count    
    replacement_char = 'L' if uppercase_count > lowercase_count else 'l'
    modified_string = re.sub(r'(?<=[a-zA-Z])1', replacement_char, input_string)
    return modified_string

def find_special_price_label(ocr_strings, label_dict, max_distance_percentage=0.2):
    label_counts = {label: 0 for label in label_dict.keys()}
    preprocessed_label_names = []
    for label, signal_words in label_dict.items():
        for word in signal_words:
            preprocessed_label_names.append((label, preprocess_string(word)))

    for line in ocr_strings:
        normalized_line = preprocess_string(line)

        for original_label_name, preprocessed_signal_word in preprocessed_label_names:
            signal_word_len = len(preprocessed_signal_word)
            max_distance = int(signal_word_len * max_distance_percentage)

            distance = Levenshtein.distance(preprocessed_signal_word, normalized_line)
            if distance <= max_distance:
                label_counts[original_label_name] += 1
                break  

            for i in range(len(normalized_line) - signal_word_len + 1):
                substring = normalized_line[i:i + signal_word_len]
                distance = Levenshtein.distance(preprocessed_signal_word, substring)

                if distance <= max_distance:
                    label_counts[original_label_name] += 1
                    break  
    return label_counts

def get_ocr_item_above(ocr_coordinates, ocr_lilt_list):
    target_x1, target_y1, target_x2, target_y2 = ocr_coordinates
    target_width = target_x2 - target_x1
    column_filtered = [
        item for item in ocr_lilt_list
        if abs(item['ocr'][1][0] - target_x1) <= target_width / 2
    ]
    items_above = [
        item for item in column_filtered
        if item['ocr'][1][1] < target_y1
    ]
    items_above.sort(key=lambda x: x['ocr'][1][3], reverse=True)
    return items_above[0] if items_above else None


def get_search_area(ocr_coordinates, ocr_lilt_list):
    above_item = get_ocr_item_above(ocr_coordinates, ocr_lilt_list)
    box_height = ocr_coordinates[3] - ocr_coordinates[1]
    lower_limit = ocr_coordinates[3]

    if above_item is None:
        return lower_limit, lower_limit - (1.5 * box_height)
    else:
        upper_limit = above_item['ocr'][1][3]
        search_height = lower_limit - upper_limit
        search_height = min(search_height, 2 * box_height)
        return lower_limit, lower_limit - search_height

def filter_strings_by_alphabet(strings, threshold=0.55):
    filtered_strings = []
    for string in strings:
        if not string:  
            continue
        total_chars = len(string)
        alpha_chars = sum(char.isalpha() for char in string)
        if total_chars > 0 and (alpha_chars / total_chars) >= threshold:
            filtered_strings.append(string)
    return filtered_strings



def remove_total_price_row(products, total_price):
    if products is None or len(products) <= 1:
        return products
    else:
        filtered_products = []
        for product in products:
            product_price = float(product['price']) if product['price'] is not None else None
            total_price_float = float(total_price) if total_price is not None else None
            if product_price != total_price_float:
                filtered_products.append(product)
        return filtered_products


def should_add_ml_product(products, ml_product):
    for product in products:
        for product_name in product['product']:
            if product_name in ml_product['name'] and product['price'] == ml_product['price']:
                return False
    return True


def add_missing_ml_rows(products, combined_result):
    ml_products = []
    if combined_result['receipt'] and combined_result['receipt']['item_table']:
        for product in combined_result['receipt']['item_table']:
            if product['description'] and product['description']['text']:
                product_name = product['description']['text'][0].replace('±', ' ')
                product_name = replace_1_with_case_based_l(product_name)
            else:
                product_name = None
            if product['price'] and product['price']['corrected']:
                product_price = product['price']['corrected']
                product_price = correct_price(str(product_price))
            elif product['price'] and product['price']['text']:
                product_price = product['price']['text']
                product_price = correct_price(str(product_price))
            else:
                product_price = None

            y_position = y_position = product['description']['bbox'][0][1] if product['description'] and product['description'].get('bbox') else None

            if product_name and product_price:
                ml_products.append({'name': product_name, 'price': product_price, 'y_position': y_position})


    for ml_product in ml_products:
        if should_add_ml_product(products, ml_product):
            if products is None:
                products = []
            products.append({'product': [ml_product['name']], 'price': ml_product['price'], 'y_position': ml_product['y_position']})
    if products is not None:
        products = sorted(products, key=lambda x: x['y_position'])
    return products


def clean_product_list(products, combined_result, total_price):
    products = remove_total_price_row(products, total_price)
    products = add_missing_ml_rows(products, combined_result)
    return products

def extract_products(ocr_lilt_list, image_width, image_height):
    try:
        right_side_filtered = filter_right_side(ocr_lilt_list, image_width/2)
        filtered_prices = filter_prices(right_side_filtered)
        grouped_prices = group_prices(filtered_prices)
        grouped_prices = sorted(grouped_prices, key=lambda group: group[0]['ocr'][1][1])
        grouped_prices = remove_duplicates(grouped_prices)

        max_count = 0
        group_with_most_unit_price = None
        for group in grouped_prices:
            count = count_unit_price_labels(group)
            if count > max_count:
                max_count = count
                group_with_most_unit_price = group

        total_deduced_price = 0
        products = []

        if group_with_most_unit_price:
            selected_price_group = group_with_most_unit_price
        else:
            # selecting the topmost-right group as alternative
            topmost_group = min(grouped_prices, key=lambda group: min(entry['ocr'][1][1] for entry in group))
            rightmost_entry = max(topmost_group, key=lambda entry: entry['ocr'][1][0])  
            selected_group = [group for group in grouped_prices if rightmost_entry in group][0]
            selected_price_group = selected_group

        for entry in selected_price_group: 
            ocr = entry['ocr']
            lilt_info = entry.get('lilt', [])
            corrected_price = correct_price(ocr[0])
            ocr_coordinates = ocr[1]
            lower_limit, upper_limit = get_search_area(ocr_coordinates, ocr_lilt_list)
            middle_y_position = (ocr_coordinates[1] + ocr_coordinates[3]) / 2
            items_besides = filter_by_middle_and_right(ocr_lilt_list, upper_limit, lower_limit, ocr[1][0])
            items_besides.sort(key=lambda x: x['ocr'][1][1])
            labels = [x['ocr'][0].replace('±', ' ') for x in items_besides]
            special_labels = find_special_price_label(labels, special_store_label)

            if any(value > 0 for value in special_labels.values()):
                pass
            else:
                if len(labels) > 0:
                    product_text_list = [replace_1_with_case_based_l(replace_1_with_case_based_l(x) )for x in filter_strings_by_alphabet(labels)]
                    product_text_list = list(set(product_text_list))
                    product_text_list = [x for x in product_text_list if x not in remove_labels]
                    total_deduced_price+= float(corrected_price)
                    products.append({'product': product_text_list, 'price': corrected_price, 'y_position': middle_y_position})
        return products
    except Exception as e:
        return None
        