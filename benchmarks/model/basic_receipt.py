from dataclasses import dataclass
from typing import List

@dataclass
class Article():
    description: str
    quantity: float
    tot_art_price: float
    unit_price: float

@dataclass
class Receipt():
    vendor: str
    address: str
    date: str
    articles: List[Article]
    total: float