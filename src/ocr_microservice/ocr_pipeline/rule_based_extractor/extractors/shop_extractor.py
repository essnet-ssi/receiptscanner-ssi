import re
import Levenshtein

store_dict = {
        "Action": ["Action"],
        "Albert Heijn": ["Albert Heijn", "AH Bonus", "BONUSKAART", "AH TO GO", "BONUS BOX", 'KOOPZEGELS', 'AH MILES'],
        "Aldi": ["Aldi"],
        "Auchan": ["Auchan"],
        "Bristol": ["Bristol"],
        "Blokker": ["blokker"],
        "Bike Totaal": ["Bike Totaal"],
        "Big Bazar": ["Big Bazar"],
        "Bruna": ["Bruna", "www.bruna.nl"],
        "Blie": ["bliewinkel"],
        "Bijenkorf": ["Bijenkorf"],
        "Burger King": ["Burger King", "Whopper"],
        # "C&A": ["C&A"], # Too short, too many false positives
        "Cactus": ["Cactus"],
        "Colruyt": ["Colruyt"],
        "Coop": ["Coop"],
        "DekaMarkt": ["DekaMarkt"],
        "Delhaize": ["Delhaize"],
        "Decathlon": ["Decathlon"],
        "Dominos": ["dominos.nl"],
        "Dirk": ["Dirk", "Dirk van den Broek"],
        "Ekoplaza": ["Ekoplaza"],
        "Esso": ["Esso", "Ongelood"],
        "Etos": ["Etos", "beste drogist met", "Wat je vraag ook is"],
        "Gall & Gall": ["Gall&Gall", "Gall & Gall"],
        "Gamma": ["Gamma"],
        "H&M": ["Hennes&Mauritz"],
        "Hema": ["Hema", "Hema.nl"],
        "Holland&Barret": ["hollandandbarrett", "hollandandbarrett.nl"],
        "Hornbach": ["Hornbach"],
        "Hoogvliet": ["Hoogvliet"],
        "Intratuin": ["Intratuin"],
        "Intertoys": ["Intertoys"],
        "Ikea": ["Ikea", "IKEA.nl"],
        "JanLinders": ["JanLinders"],
        "Jumbo": ["Jumbo", "www.jumbo.com"],
        "Kruidvat": ["Kruidvat"],
        "Kwantum": ["Kwantum"],
        "Lidl": ["Lidl"],
        "McDonald's": ["McDonald"],
        "Mediamarkt": ["Mediamarkt"],
        "New Yorker": ["New Yorker"],
        "Op = Op": ["Op=Op", "Op = Op"],
        "Plus": ["Daar houden we van", "facebook.com/plus"],
        "Praxis": ["Praxis"],
        "Primera": ["Primera"],
        "Rituals": ["Rituals"],
        "Sma+ch": ["sma+ch"],
        "Scapino": ["Scapino"],
        "Shell": ["Shell", "EUR/Liter"],
        "Sissy-Boy": ["Sissy-Boy"],
        "Sate Lounge ": ["satelounge"],
        "Trekpleister": ["Trekpleister"],
        "vanHaren": ["vanHaren"],
        "Vitaminstore": ["Vitaminstore"],
        "Vomar": ["Vomar"],
        "Wibra": ["Wibra"],
        "Xenos": ["Xenos"],
        "Zeeman": ["Zeeman"]
        
    }


def preprocess_string(s):
    return re.sub(r'[^a-z0-9 &+-]', '', s.lower().replace('±', ' '))


def find_store_counts(ocr_strings, store_dict, max_distance_percentage=0.2):
    store_counts = {store: 0 for store in store_dict.keys()}
    preprocessed_store_names = []
    for store, signal_words in store_dict.items():
        for word in signal_words:
            preprocessed_store_names.append((store, preprocess_string(word)))

    for line in ocr_strings:
        normalized_line = preprocess_string(line)
        for original_store_name, preprocessed_signal_word in preprocessed_store_names:
            signal_word_len = len(preprocessed_signal_word)
            max_distance = int(signal_word_len * max_distance_percentage)
            for i in range(len(normalized_line) - signal_word_len + 1):
                substring = normalized_line[i:i + signal_word_len]
                if Levenshtein.distance(preprocessed_signal_word, substring) <= max_distance:
                    store_counts[original_store_name] += 1
                    break 
    return store_counts


def extract_shop_name(ocr_strings):
    store_counts = find_store_counts(ocr_strings, store_dict)
    max_store = max(store_counts, key=store_counts.get)
    if store_counts[max_store] > 0:
        return max_store
    else:
        return None
