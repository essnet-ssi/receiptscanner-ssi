import re
from collections import defaultdict
from datetime import datetime

def extract_dates(ocr_strings):
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',   # e.g., 01/02/2024
        r'\d{1,2}-\d{1,2}-\d{2,4}',   # e.g., 01-02-2024
        r'\d{4}/\d{1,2}/\d{1,2}',     # e.g., 2024/01/02
        r'\d{1,2}\.\d{1,2}\.\d{2,4}', # e.g., 01.02.2024
    ]
    combined_pattern = '|'.join(date_patterns)
    date_counts = defaultdict(int)

    for line in ocr_strings:
        potential_matches = re.findall(r'.*?(\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|\d{4}/\d{1,2}/\d{1,2}|\d{1,2}\.\d{1,2}\.\d{2,4}).*?', line)
        for match in potential_matches:
            for date_format in ["%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%Y/%m/%d", "%d.%m.%Y", "%d.%m.%y"]:
                try:
                    parsed_date = datetime.strptime(match, date_format)
                    normalized_date = parsed_date.strftime("%Y-%m-%d")
                    date_counts[normalized_date] += 1
                    break
                except ValueError:
                    continue
    return date_counts

def extract_date(ocr_strings):
    date_counts = extract_dates(ocr_strings)
    if not date_counts:
        return None
    most_frequent_date = max(date_counts, key=date_counts.get)
    most_frequent_count = date_counts[most_frequent_date]
    if most_frequent_count > 0:
        return most_frequent_date
    else:
        return None