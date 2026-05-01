"""
=============================================================
ORCHESTRATEUR A2A (Agent-to-Agent)
=============================================================

L'orchestrateur est le "chef d'orchestre".
Il ne scrape pas lui-même. Il coordonne les agents.

Son rôle :
1. Créer un agent pour chaque site
2. Lancer tous les agents un par un
3. Récupérer les données de chaque agent
4. Fusionner toutes les données en un seul tableau
5. Sauvegarder le résultat dans un fichier CSV

Pourquoi "A2A" ?
→ Agent-to-Agent signifie que les agents sont indépendants.
   Chaque agent sait scraper UN type de site.
   L'orchestrateur les fait travailler ensemble.
=============================================================
"""

import pandas as pd
import logging
import os
from scraping.shopify_agent import ShopifyAgent
from scraping.woocommerce_agent import WooCommerceAgent
from scraping.config import SHOPIFY_STORES, WOOCOMMERCE_STORES, MAX_PAGES_PER_STORE, REQUEST_DELAY, OUTPUT_FILE

logger = logging.getLogger(__name__)


class A2AOrchestrator:
    """
    Orchestrateur qui coordonne tous les agents de scraping.
    """

    def __init__(self):
        self.agents = []        # Liste de tous les agents
        self.results = []       # Résultats de chaque agent

    def setup_agents(self):
        """
        Crée un agent pour chaque site dans la configuration.

        Pour les sites Shopify → on crée un ShopifyAgent
        Pour les sites WooCommerce → on crée un WooCommerceAgent
        """
        # Création des agents Shopify
        for store in SHOPIFY_STORES:
            agent = ShopifyAgent(
                store_name=store["name"],
                store_url=store["url"],
                store_country=store["country"],
                store_category=store["category"]
            )
            self.agents.append(agent)
            logger.info(f"Agent ajouté : {store['name']} (Shopify)")

        # Création des agents WooCommerce
        for store in WOOCOMMERCE_STORES:
            agent = WooCommerceAgent(
                store_name=store["name"],
                store_url=store["url"],
                store_country=store["country"],
                store_category=store["category"]
            )
            self.agents.append(agent)
            logger.info(f"Agent ajouté : {store['name']} (WooCommerce)")

        logger.info(f"Total : {len(self.agents)} agents créés")

    def run_all(self):
        """
        Lance tous les agents et collecte les résultats.

        Chaque agent travaille indépendamment :
        1. L'agent se connecte à son site
        2. Il scrape les produits
        3. Il retourne un DataFrame
        4. L'orchestrateur récupère le résultat
        """
        self.results = []

        for i, agent in enumerate(self.agents, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Agent {i}/{len(self.agents)} : {agent.store_name}")
            logger.info(f"{'='*50}")

            try:
                df = agent.scrape_products(
                    max_pages=MAX_PAGES_PER_STORE,
                    delay=REQUEST_DELAY
                )
                if not df.empty:
                    self.results.append(df)
                    logger.info(f"✓ {len(df)} produits récupérés")
                else:
                    logger.warning(f"⚠ Aucun produit récupéré pour {agent.store_name}")
            except Exception as e:
                logger.error(f"✗ Erreur pour {agent.store_name} : {e}")

        return self.merge_results()

    def merge_results(self):
        """
        Fusionne les résultats de tous les agents en un seul DataFrame.

        C'est comme rassembler les pages de plusieurs carnets
        dans un seul grand tableau.
        """
        if not self.results:
            logger.error("Aucun résultat à fusionner !")
            return pd.DataFrame()

        merged = pd.concat(self.results, ignore_index=True)

        # Suppression des doublons (même titre + même boutique)
        before = len(merged)
        merged = merged.drop_duplicates(subset=["title", "store_name"], keep="first")
        after = len(merged)

        if before > after:
            logger.info(f"Doublons supprimés : {before - after}")

        logger.info(f"\n✓ Dataset final : {len(merged)} produits, {len(merged.columns)} colonnes")
        return merged

    def save_to_csv(self, df, filepath=None):
        """
        Sauvegarde le DataFrame dans un fichier CSV.
        """
        if filepath is None:
            filepath = OUTPUT_FILE

        # Création du dossier si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        logger.info(f"✓ Données sauvegardées dans {filepath}")
        logger.info(f"  → {len(df)} lignes, {len(df.columns)} colonnes")
