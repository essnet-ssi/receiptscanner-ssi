import Levenshtein
import re

total_price_words = ["Totaal", "Te Betalen", "Totaalprijs", "Total", "art.", "PIN", "TOTAAL", "VPAY", "Betaalwijze", "UPay"]

def preprocess_string(s):
    s = s.lower()
    s = s.replace("±", " ")
    s = " ".join(s.split())  
    return s

def is_total_price_string(string, max_distance_percentage=0.2):
    global total_price_words
    preprocessed_words = [preprocess_string(w) for w in total_price_words]
    normalized_line = preprocess_string(string)
    
    for pre_word in preprocessed_words:
        signal_word_len = len(pre_word)
        max_distance = int(signal_word_len * max_distance_percentage)
        distance = Levenshtein.distance(pre_word, normalized_line)
        if distance <= max_distance:
            return True
        for i in range(len(normalized_line) - signal_word_len + 1):
            substring = normalized_line[i:i + signal_word_len]
            distance = Levenshtein.distance(pre_word, substring)
            if distance <= max_distance:
                return True
    return False

def get_line_vertical_center(bbox):
        _, top, _, bottom = bbox
        return (top + bottom) / 2.0

def get_line_rightmost_x(bbox):
    _, _, right, _ = bbox
    return right

def extract_price(text):
    pattern = re.compile(r"(?<!\d)(\d+[.,]\d{2})(?!\d)")
    match = pattern.search(text)
    if match:
        return match.group(1).replace(",", ".")
    return None


def determine_most_likely_total_price(ocr_results):
        price_counts = {}
        for result in ocr_results:
            price = result['extracted_price']
            price_counts[price] = price_counts.get(price, 0) + 1
        
        most_common_prices = [price for price, count in price_counts.items() if count > 1]
        if most_common_prices:
            return most_common_prices[0]  
        
        ocr_results_sorted = sorted(ocr_results, key=lambda x: x['full_word_distance'])
        lowest_distance_price = ocr_results_sorted[0]['extracted_price']
        lowest_distance_score = ocr_results_sorted[0]['full_word_distance']
        
        if lowest_distance_score < float('inf'): 
            return lowest_distance_price
        
        return max(result['extracted_price'] for result in ocr_results)


def get_closest_match_score(string):
        total_price_words = ["Totaal", "Te Betalen", "Totaalprijs", "Total", "art.", "PIN", "TOTAAL", "VPAY", "Betaalwijze", "UPay"]
        preprocessed_words = [preprocess_string(w) for w in total_price_words]
        normalized_line = preprocess_string(string)
        closest_score = min(Levenshtein.distance(normalized_line, pre_word) for pre_word in preprocessed_words)
        return closest_score


def find_prices_for_total_lines(ocr_lilt_list):
    total_lines = []
    for item in ocr_lilt_list:
        text = item['ocr'][0]
        if is_total_price_string(text):
            total_lines.append(item)

    results = []
    for total_line in total_lines:
        total_text, total_bbox, _ = total_line['ocr']
        total_top = total_bbox[1]
        total_bottom = total_bbox[3]

        candidate_lines = []
        for item in ocr_lilt_list:
            text, bbox, _ = item['ocr']
            if item is total_line:
                continue

            vertical_center = get_line_vertical_center(bbox)
            if total_top <= vertical_center <= total_bottom:
                candidate_lines.append(item)

        if candidate_lines:
            candidate_lines.sort(key=lambda i: get_line_rightmost_x(i['ocr'][1]), reverse=True)
            rightmost_line = candidate_lines[0]
            results.append((total_line, rightmost_line))

    return results

def extract_total_price(ocr_lilt_list, products):
    try:
        matching_results = find_prices_for_total_lines(ocr_lilt_list)
        prices = []
        for total_line, matched_line in matching_results:
            total_text, _, _ = total_line['ocr']
            matched_text, _, _ = matched_line['ocr']
            
            extracted_price = extract_price(matched_text)
            if extracted_price is not None:
                price = float(extracted_price)
                full_word_distance = get_closest_match_score(total_text)
                prices.append({
                    "total_text": total_text,
                    "extracted_price": price,
                    "full_word_distance": full_word_distance
                })
        if prices:
            return determine_most_likely_total_price(prices)
        else:
            if not products or len(products) == 0:
                return None
            total_price = 0
            for product in products:
                total_price += float(product['price'])
            return total_price
    except Exception as e:
        print(f"Error in extract_total_price: {e}")
        return None