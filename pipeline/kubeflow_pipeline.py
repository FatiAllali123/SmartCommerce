"""
=============================================================
PIPELINE KUBEFLOW - Smart eCommerce Intelligence
=============================================================

Ce fichier definit le pipeline ML complet avec Kubeflow Pipelines SDK (kfp).

Le pipeline enchain 4 etapes :
1. Preprocessing : nettoyage des donnees brutes
2. Training : entrainement des modeles (RF + XGBoost)
3. Evaluation : calcul des metriques
4. Top-K Selection : selection des meilleurs produits

Pourquoi Kubeflow ?
→ Permet de rendre le pipeline reproductible et automatise
→ Chaque etape est un composant isole (conteneurisable)
→ Le pipeline peut etre deploie sur Kubernetes

Pour lancer en local (sans Kubernetes) :
→ python pipeline/kubeflow_pipeline.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


# ==============================================================
# COMPOSANT 1 : PREPROCESSING
# ==============================================================
def preprocess_data(input_path: str, output_path: str) -> str:
    """
    Etape 1 du pipeline : Nettoyage des donnees brutes.

    Entree : fichier CSV brut (products_raw.csv)
    Sortie : fichier CSV nettoye (products_clean.csv)

    Ce que cette etape fait :
    - Supprime les produits sans prix
    - Supprime les doublons
    - Nettoie les descriptions HTML
    - Cree de nouvelles variables (feature engineering)
    - Calcule le score de chaque produit
    - Cree la variable cible (produit_succes)
    """
    logger.info("=" * 50)
    logger.info("PIPELINE - ETAPE 1 : PREPROCESSING")
    logger.info("=" * 50)

    # Ajouter le chemin du projet au sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from ml.preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(
        input_path=input_path,
        output_path=output_path
    )
    df = preprocessor.run()

    logger.info(f"Preprocessing termine : {len(df)} produits sauvegardes dans {output_path}")
    return output_path


# ==============================================================
# COMPOSANT 2 : TRAINING
# ==============================================================
def train_models(data_path: str, model_output_dir: str) -> str:
    """
    Etape 2 du pipeline : Entrainement des modeles.

    Entree : fichier CSV nettoye
    Sortie : modeles entraines (sauvegardes en pickle)

    Modeles entraines :
    - Random Forest (classification supervisee)
    - XGBoost (classification supervisee)
    - KMeans (clustering non-supervise)
    """
    logger.info("=" * 50)
    logger.info("PIPELINE - ETAPE 2 : TRAINING")
    logger.info("=" * 50)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from xgboost import XGBClassifier

    # Charger les donnees
    df = pd.read_csv(data_path)
    logger.info(f"Donnees chargees : {len(df)} produits")

    # Variables pour la classification supervisee
    feature_cols = [
        'price', 'discount_pct', 'variants_count', 'images_count',
        'is_available', 'inventory_quantity', 'tags_count',
        'word_count_description', 'description_length',
        'title_length', 'has_discount', 'has_images',
        'has_variants', 'product_type_encoded', 'store_encoded',
        'country_encoded', 'product_age_days'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df['produit_succes']

    # Separation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Random Forest ---
    logger.info("Entrainement Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        min_samples_split=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    logger.info(f"Random Forest entraine. Score train : {rf.score(X_train, y_train):.4f}")

    # --- XGBoost ---
    logger.info("Entrainement XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100, max_depth=6,
        learning_rate=0.1, subsample=0.8,
        random_state=42, eval_metric='logloss',
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    logger.info(f"XGBoost entraine. Score train : {xgb.score(X_train, y_train):.4f}")

    # --- KMeans ---
    logger.info("Entrainement KMeans...")
    cluster_features = ['price', 'discount_pct', 'variants_count', 'images_count',
                       'is_available', 'tags_count', 'word_count_description']
    cluster_features = [c for c in cluster_features if c in df.columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_features].fillna(0))

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    logger.info("KMeans entraine avec K=6")

    # Sauvegarder les modeles
    os.makedirs(model_output_dir, exist_ok=True)

    models = {
        'random_forest': rf,
        'xgboost': xgb,
        'kmeans': kmeans,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'cluster_features': cluster_features,
        'X_test': X_test,
        'y_test': y_test
    }

    model_path = os.path.join(model_output_dir, 'models.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)

    logger.info(f"Modeles sauvegardes dans {model_path}")
    return model_path


# ==============================================================
# COMPOSANT 3 : EVALUATION
# ==============================================================
def evaluate_models(model_path: str, metrics_output_path: str) -> str:
    """
    Etape 3 du pipeline : Evaluation des modeles.

    Entree : modeles entraines
    Sortie : metriques de performance (CSV)

    Metriques calculees :
    - Accuracy, Precision, Recall, F1-score
    - Matrice de confusion
    - Silhouette score pour KMeans
    """
    logger.info("=" * 50)
    logger.info("PIPELINE - ETAPE 3 : EVALUATION")
    logger.info("=" * 50)

    import pickle
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, silhouette_score
    )

    # Charger les modeles
    with open(model_path, 'rb') as f:
        models = pickle.load(f)

    rf = models['random_forest']
    xgb = models['xgboost']
    kmeans = models['kmeans']
    scaler = models['scaler']
    X_test = models['X_test']
    y_test = models['y_test']
    cluster_features = models['cluster_features']

    results = []

    # --- Evaluation Random Forest ---
    logger.info("Evaluation Random Forest...")
    y_pred_rf = rf.predict(X_test)
    rf_metrics = {
        'Modele': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf),
        'Recall': recall_score(y_test, y_pred_rf),
        'F1-score': f1_score(y_test, y_pred_rf),
        'Silhouette': None
    }
    results.append(rf_metrics)
    logger.info(f"  RF - Accuracy: {rf_metrics['Accuracy']:.4f}, F1: {rf_metrics['F1-score']:.4f}")

    # --- Evaluation XGBoost ---
    logger.info("Evaluation XGBoost...")
    y_pred_xgb = xgb.predict(X_test)
    xgb_metrics = {
        'Modele': 'XGBoost',
        'Accuracy': accuracy_score(y_test, y_pred_xgb),
        'Precision': precision_score(y_test, y_pred_xgb),
        'Recall': recall_score(y_test, y_pred_xgb),
        'F1-score': f1_score(y_test, y_pred_xgb),
        'Silhouette': None
    }
    results.append(xgb_metrics)
    logger.info(f"  XGB - Accuracy: {xgb_metrics['Accuracy']:.4f}, F1: {xgb_metrics['F1-score']:.4f}")

    # --- Evaluation KMeans ---
    logger.info("Evaluation KMeans...")
    # Recalculer silhouette sur les donnees d'entrainement
    df = pd.read_csv("data/processed/products_clean.csv")
    X_scaled = scaler.transform(df[cluster_features].fillna(0))
    labels = kmeans.predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)

    kmeans_metrics = {
        'Modele': 'KMeans (K=6)',
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1-score': None,
        'Silhouette': sil_score
    }
    results.append(kmeans_metrics)
    logger.info(f"  KMeans - Silhouette: {sil_score:.4f}")

    # Sauvegarder les metriques
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(metrics_output_path, index=False)
    logger.info(f"Metriques sauvegardees dans {metrics_output_path}")

    return metrics_output_path


# ==============================================================
# COMPOSANT 4 : TOP-K SELECTION
# ==============================================================
def select_topk(data_path: str, model_path: str, k: int, output_path: str) -> str:
    """
    Etape 4 du pipeline : Selection des Top-K produits.

    Entree : donnees nettoyees + modeles entraines
    Sortie : tableau des Top-K produits

    Le scoring utilise le final_score calcule pendant le preprocessing.
    """
    logger.info("=" * 50)
    logger.info(f"PIPELINE - ETAPE 4 : TOP-{k} SELECTION")
    logger.info("=" * 50)

    # Charger les donnees
    df = pd.read_csv(data_path)

    # Selectionner les Top-K
    topk = df.nlargest(k, 'final_score')

    # Colonnes a afficher
    display_cols = ['title', 'price', 'store_name', 'product_type',
                    'discount_pct', 'available', 'final_score', 'produit_succes']
    display_cols = [c for c in display_cols if c in topk.columns]

    topk = topk[display_cols]

    # Sauvegarder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    topk.to_csv(output_path, index=False)

    logger.info(f"Top-{k} produits sauvegardes dans {output_path}")
    logger.info(f"\n{topk.to_string(index=False)}")

    return output_path


# ==============================================================
# DEFINITION DU PIPELINE
# ==============================================================
def run_pipeline():
    """
    Execute le pipeline complet.

    Enchainement :
    1. Preprocessing (nettoyage)
    2. Training (entrainement modeles)
    3. Evaluation (metriques)
    4. Top-K Selection (meilleurs produits)

    Chaque etape recoit les sorties de l'etape precedente.
    """
    print("=" * 60)
    print("  PIPELINE KUBEFLOW - Smart eCommerce Intelligence")
    print(f"  Demare le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Chemins
    RAW_DATA = "data/raw/products_raw.csv"
    CLEAN_DATA = "data/processed/products_clean.csv"
    MODEL_DIR = "outputs/models"
    METRICS_PATH = "outputs/pipeline_metrics.csv"
    TOPK_PATH = "outputs/top_k_products.csv"
    K = 20

    # ETAPE 1 : Preprocessing
    print("\n  [1/4] Preprocessing...")
    preprocess_data(RAW_DATA, CLEAN_DATA)

    # ETAPE 2 : Training
    print("\n  [2/4] Training...")
    model_path = train_models(CLEAN_DATA, MODEL_DIR)

    # ETAPE 3 : Evaluation
    print("\n  [3/4] Evaluation...")
    evaluate_models(model_path, METRICS_PATH)

    # ETAPE 4 : Top-K Selection
    print("\n  [4/4] Top-K Selection...")
    select_topk(CLEAN_DATA, model_path, K, TOPK_PATH)

    # Resume
    print("\n" + "=" * 60)
    print("  PIPELINE TERMINE AVEC SUCCES")
    print("=" * 60)
    print(f"  Donnees nettoyees : {CLEAN_DATA}")
    print(f"  Modeles entraines : {MODEL_DIR}/models.pkl")
    print(f"  Metriques         : {METRICS_PATH}")
    print(f"  Top-{K} produits   : {TOPK_PATH}")
    print("=" * 60)


# ==============================================================
# POINT D'ENTREE
# ==============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    run_pipeline()
