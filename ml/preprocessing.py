"""
=============================================================
NETTOYAGE ET CONSTRUCTION DU DATASET
=============================================================

Ce fichier fait le lien entre le scraping (etape 1) et
l'analyse ML (etape 3+).

Il transforme les donnees brutes en donnees propres et pretes
pour le Data Mining.

Etapes :
1. Charger les donnees brutes
2. Explorer (voir ce qu'on a)
3. Nettoyer (supprimer les problemes)
4. Creer de nouvelles variables (feature engineering)
5. Calculer le score de chaque produit
6. Creer la variable cible "produit_succes"
7. Sauvegarder le dataset propre
"""

import pandas as pd
import numpy as np
import re
import os
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Classe qui nettoie et prepare les donnees produits.

    Pourquoi une classe ?
    → Pour organiser le code proprement. Chaque fonction
       a un role precis, et on les appelle dans l'ordre.
    """

    def __init__(self, input_path, output_path):
        """
        Parametres :
        - input_path : chemin du fichier brut (products_raw.csv)
        - output_path : chemin du fichier nettoye (products_clean.csv)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def run(self):
        """
        Lance tout le processus de nettoyage dans l'ordre.
        C'est la fonction principale qu'on appelle.
        """
        logger.info("=" * 50)
        logger.info("DEBUT DU NETTOYAGE DES DONNEES")
        logger.info("=" * 50)

        # Etape 1 : Charger les donnees
        self.load_data()

        # Etape 2 : Explorer les donnees
        self.explore_data()

        # Etape 3 : Nettoyer
        self.clean_data()

        # Etape 4 : Feature engineering (creer de nouvelles variables)
        self.create_features()

        # Etape 5 : Calculer le score
        self.calculate_score()

        # Etape 6 : Creer la variable cible
        self.create_target_variable()

        # Etape 7 : Sauvegarder
        self.save_data()

        return self.df

    # =========================================================
    # ETAPE 1 : CHARGER LES DONNEES
    # =========================================================
    def load_data(self):
        """
        Charge le fichier CSV brut dans un DataFrame pandas.

        Un DataFrame, c'est quoi ?
        → C'est un tableau, comme dans Excel, mais en Python.
           Chaque ligne = un produit
           Chaque colonne = une information (prix, titre, etc.)
        """
        logger.info(f"Chargement de {self.input_path}...")

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Fichier non trouve : {self.input_path}")

        self.df = pd.read_csv(self.input_path)
        logger.info(f"Charge : {len(self.df)} produits, {len(self.df.columns)} colonnes")

    # =========================================================
    # ETAPE 2 : EXPLORER LES DONNEES
    # =========================================================
    def explore_data(self):
        """
        Affiche un resume des donnees pour comprendre ce qu'on a.

        C'est comme ouvrir une boite et regarder ce qu'il y a dedans
        avant de commencer a trier.
        """
        df = self.df

        logger.info("\n--- EXPLORATION DES DONNEES ---")
        logger.info(f"Nombre de produits : {len(df)}")
        logger.info(f"Nombre de colonnes : {len(df.columns)}")
        logger.info(f"Colonnes : {list(df.columns)}")

        # Valeurs manquantes par colonne
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        logger.info(f"\nValeurs manquantes :")
        for col in df.columns:
            if missing[col] > 0:
                logger.info(f"  {col} : {missing[col]} ({missing_pct[col]}%)")

        # Prix
        logger.info(f"\nStatistiques prix :")
        logger.info(f"  Min    : {df['price'].min()}")
        logger.info(f"  Max    : {df['price'].max()}")
        logger.info(f"  Moyen  : {df['price'].mean():.2f}")
        logger.info(f"  Median : {df['price'].median():.2f}")

        # Boutiques
        logger.info(f"\nProduits par boutique :")
        for store, count in df['store_name'].value_counts().items():
            logger.info(f"  {store} : {count}")

        # Categories
        logger.info(f"\nCategories de produits :")
        for cat, count in df['product_type'].value_counts().head(10).items():
            logger.info(f"  {cat} : {count}")

    # =========================================================
    # ETAPE 3 : NETTOYER
    # =========================================================
    def clean_data(self):
        """
        Supprime les problemes dans les donnees.

        Qu'est-ce qu'on nettoie ?
        1. Les produits sans prix (prix = 0 ou NaN)
        2. Les doublons
        3. Les descriptions HTML (on enleve les balises)
        4. Les prix negatifs ou aberrants
        """
        df = self.df
        initial_count = len(df)

        # --- 3.1 : Supprimer les produits sans prix ---
        # Un produit sans prix n'est pas analysable
        df = df[df['price'] > 0].copy()
        logger.info(f"Produits avec prix > 0 : {len(df)} (supprime {initial_count - len(df)})")

        # --- 3.2 : Supprimer les doublons ---
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['product_id'], keep='first')
        logger.info(f"Apres suppression doublons : {len(df)} (supprime {before_dedup - len(df)})")

        # --- 3.3 : Nettoyer les descriptions HTML ---
        # Les descriptions contiennent des balises HTML comme <p>, <br>, <strong>
        # On les enleve pour garder juste le texte
        df['description_clean'] = df['description'].apply(self._clean_html)
        df['description_length'] = df['description_clean'].apply(len)

        # --- 3.4 : Nettoyer les titres ---
        df['title_clean'] = df['title'].str.strip()
        df['title_length'] = df['title_clean'].apply(len)

        # --- 3.5 : Gérer les valeurs manquantes ---
        # Pour les colonnes numeriques : remplacer NaN par 0
        numeric_cols = ['discount_pct', 'inventory_quantity', 'variants_count', 'images_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Pour les colonnes texte : remplacer NaN par ""
        text_cols = ['tags', 'product_type', 'variant_options']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("")

        # --- 3.6 : Supprimer les prix aberrants ---
        # On garde les produits entre 0.5 et 5000 euros
        # (en dessous = probablement une erreur, au-dessus = probablement un pack)
        before_price = len(df)
        df = df[(df['price'] >= 0.5) & (df['price'] <= 5000)].copy()
        logger.info(f"Apres filtrage prix (0.5 - 5000) : {len(df)} (supprime {before_price - len(df)})")

        self.df = df
        logger.info(f"\nNettoyage termine : {initial_count} → {len(df)} produits")

    @staticmethod
    def _clean_html(text):
        """
        Enleve les balises HTML d'un texte.

        Exemple :
        "<p>Super <strong>casque</strong> bluetooth</p>"
        → "Super casque bluetooth"
        """
        if pd.isna(text) or text == "":
            return ""
        # Enlever les balises HTML
        clean = re.sub(r'<[^>]+>', ' ', str(text))
        # Enlever les espaces multiples
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    # =========================================================
    # ETAPE 4 : FEATURE ENGINEERING
    # =========================================================
    def create_features(self):
        """
        Cree de nouvelles variables a partir des donnees existantes.

        Pourquoi ?
        → Les algorithmes ML ont besoin de chiffres pour travailler.
           On transforme les informations textuelles en variables
           numeriques exploitables.

        Quelles variables on cree ?
        1. has_discount : est-ce que le produit a une remise ? (0 ou 1)
        2. has_images : est-ce que le produit a des images ? (0 ou 1)
        3. has_variants : est-ce que le produit a des variantes ? (0 ou 1)
        4. tags_count : combien de tags le produit a
        5. word_count_description : nombre de mots dans la description
        6. price_category : categorie de prix (low, medium, high, premium)
        7. is_available : disponibilite (0 ou 1)
        """
        df = self.df
        logger.info("\n--- FEATURE ENGINEERING ---")

        # 4.1 : Le produit a-t-il une remise ?
        df['has_discount'] = (df['discount_pct'] > 0).astype(int)

        # 4.2 : Le produit a-t-il des images ?
        df['has_images'] = (df['images_count'] > 0).astype(int)

        # 4.3 : Le produit a-t-il des variantes ?
        df['has_variants'] = (df['variants_count'] > 1).astype(int)

        # 4.4 : Nombre de tags
        df['tags_count'] = df['tags'].apply(
            lambda x: len(str(x).split(',')) if str(x).strip() else 0
        )

        # 4.5 : Nombre de mots dans la description
        df['word_count_description'] = df['description_clean'].apply(
            lambda x: len(str(x).split()) if str(x).strip() else 0
        )

        # 4.6 : Categorie de prix
        # On decoupe les prix en 4 categories
        df['price_category'] = pd.cut(
            df['price'],
            bins=[0, 20, 50, 100, 5000],
            labels=['low', 'medium', 'high', 'premium'],
            include_lowest=True
        )

        # 4.7 : Disponibilite (convertir en 0/1)
        df['is_available'] = df['available'].astype(int)

        # 4.8 : Age du produit en jours (depuis la creation)
        if 'created_at' in df.columns:
            # On parse les dates en ignorant les fuseaux horaires
           
            # (les dates Shopify ont un fuseau, l'heure locale n'en a pas)
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
            today = pd.Timestamp.now(tz='UTC')
            df['product_age_days'] = (today - df['created_at']).dt.days
            df['product_age_days'] = df['product_age_days'].fillna(0).astype(int)

        # 4.9 : Encodage de la categorie de produit en nombre
        # (les algorithmes ML ne comprennent pas le texte)
        df['product_type_encoded'] = df['product_type'].astype('category').cat.codes

        # 4.10 : Encodage de la boutique
        df['store_encoded'] = df['store_name'].astype('category').cat.codes

        # 4.11 : Encodage du pays
        df['country_encoded'] = df['store_country'].astype('category').cat.codes

        logger.info(f"Nouvelles variables creees : has_discount, has_images, has_variants,")
        logger.info(f"  tags_count, word_count_description, price_category, is_available,")
        logger.info(f"  product_age_days, product_type_encoded, store_encoded, country_encoded")
        logger.info(f"Total colonnes : {len(df.columns)}")

        self.df = df

    # =========================================================
    # ETAPE 5 : CALCULER LE SCORE
    # =========================================================
    def calculate_score(self):
        """
        Attribue un score synthetique a chaque produit.

        Le score combine plusieurs criteres :
        - Prix (un prix raisonnable est mieux)
        - Disponibilite (en stock = mieux)
        - Nombre de variantes (plus de choix = mieux)
        - Nombre d'images (meilleure presentation)
        - Qualite de la description (description complete)
        - Remise (une remise attire les clients)

        Chaque critere est normalise entre 0 et 1,
        puis on fait une moyenne ponderee.
        """
        df = self.df
        logger.info("\n--- CALCUL DU SCORE ---")

        # Normalisation Min-Max : transforme toute valeur en 0-1
        # Formule : (valeur - min) / (max - min)
        def normalize(series):
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - min_val) / (max_val - min_val)

        # Score du prix : un prix moyen est preferable
        # (pas trop cher, pas trop bon marche = meilleure qualite percue)
        price_norm = normalize(df['price'])
        # On inverse : un prix tres eleve = score bas, prix moyen = score haut
        df['score_price'] = 1 - abs(price_norm - 0.5) * 2
        df['score_price'] = df['score_price'].clip(0, 1)

        # Score de disponibilite
        df['score_availability'] = df['is_available'].astype(float)

        # Score des variantes : plus de variantes = plus de choix
        df['score_variants'] = normalize(df['variants_count'])

        # Score des images : plus d'images = meilleure presentation
        df['score_images'] = normalize(df['images_count'])

        # Score de la description : description complete
        df['score_description'] = normalize(df['word_count_description'])

        # Score de la remise : une remise attire
        df['score_discount'] = normalize(df['discount_pct'])

        # --- SCORE FINAL : Moyenne ponderee ---
        # Poids : on donne plus d'importance a certains criteres
        weights = {
            'score_price': 0.20,
            'score_availability': 0.25,
            'score_variants': 0.10,
            'score_images': 0.10,
            'score_description': 0.15,
            'score_discount': 0.20
        }

        df['final_score'] = 0
        for col, weight in weights.items():
            df['final_score'] += df[col] * weight

        # Arrondir a 3 decimales
        df['final_score'] = df['final_score'].round(3)

        logger.info(f"Score calcule pour {len(df)} produits")
        logger.info(f"Score min    : {df['final_score'].min()}")
        logger.info(f"Score max    : {df['final_score'].max()}")
        logger.info(f"Score moyen  : {df['final_score'].mean():.3f}")

        self.df = df

    # =========================================================
    # ETAPE 6 : CREER LA VARIABLE CIBLE
    # =========================================================
    def create_target_variable(self):
        """
        Cree la variable cible "produit_succes".

        C'est quoi une variable cible ?
        → C'est ce qu'on veut predire. Ici, on veut predire
           si un produit va etre un "succes" ou non.

        Comment on definit "succes" ?
        → Les 20% de produits ayant le meilleur score
           sont consideres comme des succes (produit_succes = 1)
        → Les 80% restants ne sont pas des succes (produit_succes = 0)

        Pourquoi 20% ?
        → C'est un seuil classique en analyse business.
           On considere que le top 20% represente les "meilleurs" produits.
        """
        df = self.df
        logger.info("\n--- VARIABLE CIBLE ---")

        # Calcul du seuil (80e percentile)
        threshold = df['final_score'].quantile(0.80)
        logger.info(f"Seuil Top 20% : {threshold:.3f}")

        # Creation de la variable cible
        df['produit_succes'] = (df['final_score'] >= threshold).astype(int)

        # Verification
        count_success = df['produit_succes'].sum()
        count_total = len(df)
        logger.info(f"Produits succes   : {count_success} ({count_success/count_total*100:.1f}%)")
        logger.info(f"Produits non succes : {count_total - count_success} ({(count_total-count_success)/count_total*100:.1f}%)")

        # Top-K produits
        logger.info(f"\n--- TOP 10 PRODUITS ---")
        top10 = df.nlargest(10, 'final_score')[['title', 'price', 'store_name', 'final_score', 'produit_succes']]
        for i, row in top10.iterrows():
            logger.info(f"  {row['title'][:50]:50s} | {row['price']:>8.2f} | {row['store_name']:15s} | Score: {row['final_score']:.3f}")

        self.df = df

    # =========================================================
    # ETAPE 7 : SAUVEGARDER
    # =========================================================
    def save_data(self):
        """
        Sauvegarde le dataset nettoye dans un fichier CSV.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        logger.info(f"\nDataset sauvegarde : {self.output_path}")
        logger.info(f"  Lignes   : {len(self.df)}")
        logger.info(f"  Colonnes : {len(self.df.columns)}")
        logger.info(f"  Colonnes : {list(self.df.columns)}")
