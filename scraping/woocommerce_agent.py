"""
=============================================================
AGENT DE SCRAPING WOOCOMMERCE
=============================================================

WooCommerce est un autre système de boutique en ligne.
Contrairement à Shopify, il n'a pas toujours d'API JSON facile.
On utilise donc deux approches :
1. D'abord, essayer l'API REST WooCommerce
2. Si ça ne marche pas, scraper le HTML avec BeautifulSoup

BeautifulSoup, c'est quoi ?
→ C'est un outil qui lit le code HTML d'une page web
   et nous permet de chercher des éléments spécifiques
   (comme chercher le prix dans une balise <span class="price">)
=============================================================
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import logging

logger = logging.getLogger(__name__)


class WooCommerceAgent:
    """
    Agent autonome de scraping pour les boutiques WooCommerce.

    Fonctionne en deux modes :
    - Mode API : utilise l'API REST de WooCommerce (si disponible)
    - Mode HTML : scrape directement la page web (fallback)
    """

    def __init__(self, store_name, store_url, store_country, store_category):
        self.store_name = store_name
        self.store_url = store_url
        self.store_country = store_country
        self.store_category = store_category
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        logger.info(f"Agent WooCommerce créé pour {store_name}")

    def scrape_products(self, max_pages=10, delay=2):
        """
        Tente de scraper les produits.
        Essaie d'abord l'API, puis le HTML si ça échoue.
        """
        # Tentative 1 : API REST WooCommerce
        logger.info(f"Tentative API REST pour {self.store_name}...")
        products = self._try_api(max_pages, delay)

        if products:
            logger.info(f"✓ {self.store_name} : {len(products)} produits via API")
            return pd.DataFrame(products)

        # Tentative 2 : Scraping HTML
        logger.info(f"API non disponible → Scraping HTML pour {self.store_name}...")
        products = self._try_html_scraping(max_pages, delay)

        logger.info(f"✓ {self.store_name} : {len(products)} produits via HTML")
        return pd.DataFrame(products)

    def _try_api(self, max_pages, delay):
        """
        Essaie de récupérer les produits via l'API REST WooCommerce.
        L'URL typique : https://site.com/wp-json/wc/store/products
        """
        all_products = []

        for page in range(1, max_pages + 1):
            url = f"https://{self.store_url}/wp-json/wc/store/products?page={page}&per_page=100"
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                if response.status_code != 200:
                    break

                products = response.json()
                if not products:
                    break

                for p in products:
                    all_products.append(self._extract_from_api(p))

                time.sleep(delay)

            except Exception as e:
                logger.warning(f"API échouée : {e}")
                return []

        return all_products

    def _try_html_scraping(self, max_pages, delay):
        """
        Scrape les données directement depuis le HTML de la page.
        C'est la méthode "classique" : on lit la page comme un navigateur.

        Comment ça marche ?
        1. On télécharge le code HTML de la page
        2. On utilise BeautifulSoup pour le "lire"
        3. On cherche les balises qui contiennent les infos produits
           (ex: <h2 class="product-title"> contient le nom du produit)
        """
        all_products = []

        for page in range(1, max_pages + 1):
            url = f"https://{self.store_url}/shop/page/{page}/"
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                if response.status_code != 200:
                    break

                soup = BeautifulSoup(response.text, "lxml")

                # On cherche tous les blocs de produits sur la page
                # WooCommerce utilise souvent la classe "product" ou "woocommerce-LoopProduct"
                product_cards = soup.select("li.product, div.product")

                if not product_cards:
                    break

                for card in product_cards:
                    product = self._extract_from_html(card)
                    all_products.append(product)

                logger.info(f"  → Page {page} : {len(product_cards)} produits")
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Erreur HTML : {e}")
                break

        return all_products

    def _extract_from_api(self, product):
        """Extraction depuis l'API REST WooCommerce"""
        prices = product.get("prices", {})
        return {
            "product_id": product.get("id"),
            "title": product.get("name", ""),
            "description": product.get("short_description", ""),
            "vendor": self.store_name,
            "product_type": product.get("type", ""),
            "tags": ", ".join([t.get("name", "") for t in product.get("tags", [])]),
            "created_at": "",
            "updated_at": "",
            "price": int(prices.get("price", 0)) / 100 if prices.get("price") else 0,
            "compare_at_price": int(prices.get("regular_price", 0)) / 100 if prices.get("regular_price") else None,
            "discount_pct": 0,
            "available": product.get("is_purchasable", False),
            "inventory_quantity": 0,
            "variants_count": len(product.get("variations", [])),
            "variant_options": "",
            "images_count": len(product.get("images", [])),
            "store_name": self.store_name,
            "store_url": self.store_url,
            "store_country": self.store_country,
            "store_category": self.store_category
        }

    def _extract_from_html(self, card):
        """
        Extraction depuis le HTML.

        On cherche les balises HTML qui contiennent les infos.
        Les sélecteurs CSS peuvent varier d'un site à l'autre,
        d'où les multiples tentatives (try/except).
        """
        # Titre du produit
        title_el = card.select_one("h2, h3, .product-title, .woocommerce-loop-product__title")
        title = title_el.get_text(strip=True) if title_el else ""

        # Prix
        price_el = card.select_one(".price .amount, .price, .woocommerce-Price-amount")
        price_text = price_el.get_text(strip=True) if price_el else "0"
        price = self._parse_price(price_text)

        # Lien du produit
        link_el = card.select_one("a[href]")
        link = link_el["href"] if link_el else ""

        # Image
        img_el = card.select_one("img")
        has_image = img_el is not None

        return {
            "product_id": hash(title + self.store_name),  # ID généré à partir du titre
            "title": title,
            "description": "",
            "vendor": self.store_name,
            "product_type": "",
            "tags": "",
            "created_at": "",
            "updated_at": "",
            "price": price,
            "compare_at_price": None,
            "discount_pct": 0,
            "available": True,
            "inventory_quantity": 0,
            "variants_count": 1,
            "variant_options": "",
            "images_count": 1 if has_image else 0,
            "store_name": self.store_name,
            "store_url": self.store_url,
            "store_country": self.store_country,
            "store_category": self.store_category,
            "product_url": link
        }

    @staticmethod
    def _parse_price(price_text):
        """Convertit un texte de prix en nombre. Ex: '29.99€' → 29.99"""
        import re
        match = re.search(r'[\d]+[.,]?[\d]*', price_text.replace(",", "."))
        if match:
            return float(match.group())
        return 0.0
