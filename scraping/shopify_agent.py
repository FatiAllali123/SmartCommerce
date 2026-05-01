"""
=============================================================
AGENT DE SCRAPING SHOPIFY
=============================================================

Qu'est-ce qu'un "agent" ?
→ C'est un composant logiciel autonome qui sait :
   1. Se connecter à un site Shopify
   2. Demander la liste des produits (via l'API JSON)
   3. Lire les données de chaque produit
   4. Les organiser dans un tableau propre

Pourquoi "A2A" (Agent-to-Agent) ?
→ Parce qu'on crée plusieurs agents (un par type de site),
   et un orchestrateur qui les coordonne. Les agents
   communiquent entre eux via l'orchestrateur.

Comment ça marche, étape par étape :
→ On envoie une requête HTTP à l'adresse du site
→ Le site répond avec du JSON (données structurées)
→ On lit le JSON et on extrait les champs qui nous intéressent
→ On recommence pour la page suivante, jusqu'à épuisement
=============================================================
"""

import requests
import pandas as pd
import time
import logging

# Configuration du système de logs
# Les logs permettent de voir ce que fait le programme pendant l'exécution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ShopifyAgent:
    """
    Agent autonome de scraping pour les boutiques Shopify.

    Ce qu'il fait concrètement :
    1. Prend en entrée l'URL d'une boutique Shopify
    2. Accède à l'adresse https://URL/products.json
    3. Lit les produits page par page (250 par page)
    4. Extrait les informations importantes de chaque produit
    5. Retourne un DataFrame pandas (tableau de données)
    """

    def __init__(self, store_name, store_url, store_country, store_category):
        """
        Initialisation de l'agent.

        Quand on crée un agent, on lui donne :
        - store_name : le nom de la boutique (ex: "Gymshark")
        - store_url : l'adresse du site (ex: "gymshark.com")
        - store_country : le pays d'origine
        - store_category : la catégorie de produits

        On construit aussi l'URL de l'API :
        → https://gymshark.com/products.json
        """
        self.store_name = store_name
        self.store_url = store_url
        self.store_country = store_country
        self.store_category = store_category
        self.api_url = f"https://{store_url}/products.json"

        # On configure les headers (en-têtes) de la requête
        # C'est comme s'identifier quand on frappe à la porte du site
        # User-Agent dit au site qu'on est un navigateur normal
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        logger.info(f"Agent créé pour {store_name} ({self.api_url})")

    def scrape_products(self, max_pages=10, delay=2):
        """
        Fonction principale : scrape tous les produits de la boutique.

        Paramètres :
        - max_pages : nombre maximum de pages à lire (pour ne pas scraper indéfiniment)
        - delay : temps d'attente entre chaque page (politesse)

        Retourne :
        - Un DataFrame pandas avec tous les produits trouvés
        """
        all_products = []    # Liste vide qui va contenir tous les produits
        page = 1             # On commence à la page 1

        logger.info(f"Début du scraping de {self.store_name}...")

        while page <= max_pages:
            # Construction de l'URL avec le numéro de page
            url = f"{self.api_url}?page={page}&limit=250"
            logger.info(f"  → Page {page} : {url}")

            try:
                # On envoie la requête au site
                response = requests.get(url, headers=self.headers, timeout=30)

                # Vérification : est-ce que le site a répondu correctement ?
                if response.status_code != 200:
                    logger.warning(f"  ⚠ Statut {response.status_code} → arrêt")
                    break

                # On convertit la réponse JSON en dictionnaire Python
                data = response.json()

                # On récupère la liste des produits
                products = data.get("products", [])

                # Si la page est vide, on a fini
                if not products:
                    logger.info(f"  → Page vide → fin du scraping pour {self.store_name}")
                    break

                # Pour chaque produit, on extrait les informations qui nous intéressent
                for product in products:
                    extracted = self._extract_product_data(product)
                    all_products.append(extracted)

                logger.info(f"  → {len(products)} produits récupérés sur cette page")

                # On attend avant de demander la page suivante (politesse)
                time.sleep(delay)
                page += 1

            except requests.exceptions.RequestException as e:
                # En cas d'erreur réseau, on arrête proprement
                logger.error(f"  ✗ Erreur réseau : {e}")
                break

        logger.info(f"✓ {self.store_name} : {len(all_products)} produits au total")
        return pd.DataFrame(all_products)

    def _extract_product_data(self, product):
        """
        Extrait les données importantes d'un seul produit.

        Le JSON Shopify contient beaucoup d'infos. On choisit celles
        qui sont utiles pour notre analyse Data Mining.

        Pourquoi ce choix de variables ?
        → Ce sont celles recommandées dans le document du prof :
           données descriptives, prix, popularité, stock, variantes, vendeur
        """
        # Prix : on prend le prix de la première variante
        # (chaque produit peut avoir plusieurs variantes : taille, couleur...)
        variants = product.get("variants", [])
        if variants:
            first_variant = variants[0]
            price = float(first_variant.get("price", 0))
            compare_price = first_variant.get("compare_at_price")
            if compare_price:
                compare_price = float(compare_price)
            available = first_variant.get("available", False)
            inventory_qty = first_variant.get("inventory_quantity", 0)
        else:
            price = 0
            compare_price = None
            available = False
            inventory_qty = 0

        # Calcul de la remise (en pourcentage)
        discount = 0
        if compare_price and compare_price > price:
            discount = round(((compare_price - price) / compare_price) * 100, 1)

        # Nombre de variantes (couleurs, tailles disponibles)
        variant_options = []
        for v in variants:
            option1 = v.get("option1", "")
            option2 = v.get("option2", "")
            if option1:
                variant_options.append(option1)
            if option2:
                variant_options.append(option2)

        # On construit un dictionnaire avec toutes les infos du produit
        return {
            "product_id": product.get("id"),
            "title": product.get("title", ""),
            "description": product.get("body_html", ""),
            "vendor": product.get("vendor", ""),
            "product_type": product.get("product_type", ""),
            "tags": ", ".join(product.get("tags", [])),
            "created_at": product.get("created_at", ""),
            "updated_at": product.get("updated_at", ""),
            "price": price,
            "compare_at_price": compare_price,
            "discount_pct": discount,
            "available": available,
            "inventory_quantity": inventory_qty,
            "variants_count": len(variants),
            "variant_options": ", ".join(set(variant_options)),
            "images_count": len(product.get("images", [])),
            "store_name": self.store_name,
            "store_url": self.store_url,
            "store_country": self.store_country,
            "store_category": self.store_category
        }
