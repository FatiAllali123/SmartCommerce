"""
Configuration des boutiques e-commerce a scraper.
"""

SHOPIFY_STORES = [
    {
        "name": "Gymshark",
        "url": "gymshark.com",
        "country": "UK",
        "category": "Sportswear"
    },
    {
        "name": "Allbirds",
        "url": "allbirds.com",
        "country": "USA",
        "category": "Shoes"
    },
    {
        "name": "Brooklinen",
        "url": "brooklinen.com",
        "country": "USA",
        "category": "Home"
    },
    {
        "name": "Pura Vida Bracelets",
        "url": "puravidabracelets.com",
        "country": "USA",
        "category": "Jewelry"
    }
]

WOOCOMMERCE_STORES = []

MAX_PAGES_PER_STORE = 10
PRODUCTS_PER_PAGE = 250
REQUEST_DELAY = 2
OUTPUT_FILE = "data/raw/products_raw.csv"
