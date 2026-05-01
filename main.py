"""
Point de depart du programme.
Etape 1 : Scraping (deja fait)
Etape 2 : Nettoyage (deja fait)
Etape 3 : Modeles ML (deja fait)
Etape 4 : Dashboard BI (lance separement)
Etape 5 : Module LLM
"""

import logging
import pandas as pd
from ml.models import (
    RandomForestModel, XGBoostModel, KMeansModel,
    DBSCANModel, PCAModel, AssociationRulesMiner
)
from ml.evaluation import ModelVisualizer
from llm.enrichment import LLMEnricher, generate_full_report

from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_ml_pipeline():
    """Lance tous les modeles ML."""
    print("=" * 60)
    print("  ETAPE 3 : Modeles Machine Learning")
    print("=" * 60)

    df = pd.read_csv("data/processed/products_clean.csv")
    print(f"\n  Dataset charge : {len(df)} produits, {len(df.columns)} colonnes")

    viz = ModelVisualizer(output_dir="outputs/figures")

    # Random Forest
    rf = RandomForestModel()
    X_train, X_test, y_train, y_test = rf.prepare_data(df)
    rf.train(X_train, y_train)
    rf_results = rf.evaluate(X_test, y_test)
    rf_importance = rf.get_feature_importance()
    viz.plot_confusion_matrix(y_test, rf_results['y_pred'], "Random Forest")
    viz.plot_feature_importance(rf_importance, "Random Forest")

    # XGBoost
    xgb = XGBoostModel()
    xgb.train(X_train, y_train, feature_names=rf.feature_names)
    xgb_results = xgb.evaluate(X_test, y_test)
    xgb_importance = xgb.get_feature_importance()
    viz.plot_confusion_matrix(y_test, xgb_results['y_pred'], "XGBoost")
    viz.plot_feature_importance(xgb_importance, "XGBoost")

    comparison = viz.generate_summary_table(rf_results, xgb_results)
    comparison.to_csv("outputs/model_comparison.csv", index=False)

    # KMeans
    from sklearn.preprocessing import StandardScaler
    cluster_features = ['price', 'discount_pct', 'variants_count', 'images_count',
                       'is_available', 'tags_count', 'word_count_description']
    cluster_features = [c for c in cluster_features if c in df.columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_features].fillna(0))

    kmeans = KMeansModel()
    best_k, silhouette_scores = kmeans.find_optimal_k(X_scaled, max_k=6)
    cluster_labels = kmeans.train(X_scaled, n_clusters=best_k)
    kmeans.interpret_clusters(df, cluster_features)
    viz.plot_silhouette_scores(range(2, 7), silhouette_scores, best_k)

    # DBSCAN
    dbscan = DBSCANModel()
    dbscan_labels = dbscan.run(X_scaled)
    anomalies = dbscan.analyze_anomalies(df)
    if len(anomalies) > 0:
        anomalies.to_csv("outputs/anomalies.csv", index=False)

    # PCA
    pca = PCAModel()
    X_pca = pca.fit_transform(X_scaled)
    viz.plot_clusters_2d(X_pca, cluster_labels, "Clusters KMeans (PCA 2D)")
    viz.plot_price_distribution(df)

    # Regles d'association
    assoc = AssociationRulesMiner()
    rules = assoc.run(df, min_support=0.05, min_confidence=0.3)
    if rules is not None and len(rules) > 0:
        rules.to_csv("outputs/association_rules.csv", index=False)

    print("\n  ML pipeline termine")


def run_llm_enrichment():
    """Lance le module LLM."""
    print("\n" + "=" * 60)
    print("  ETAPE 5 : Module LLM - Enrichissement")
    print("=" * 60)

    df = pd.read_csv("data/processed/products_clean.csv")

    # Groq API Key
    api_key = None  # Le constructeur LLMEnricher la cherche dans .env

    results = generate_full_report(df, api_key=api_key)

    # Afficher les resultats
    print("\n" + "=" * 60)
    print("  RESUME DES RAPPORTS GENERES")
    print("=" * 60)

    print("\n  1. Resumes produits :")
    print(f"     {len(results['summaries'])} resumes generes")
    print(f"     -> outputs/product_summaries.csv")

    print("\n  2. Analyse concurrentielle :")
    print(f"     -> outputs/competitive_report.txt")

    print("\n  3. Rapport de tendances :")
    print(f"     -> outputs/trend_report.txt")

    print("\n  4. Strategies marketing :")
    print(f"     -> outputs/marketing_strategies.txt")

    # Test du chatbot
    print("\n" + "=" * 60)
    print("  TEST DU CHATBOT")
    print("=" * 60)

    enricher = LLMEnricher(api_key=api_key)
    test_questions = [
        "Combien de produits avez-vous ?",
        "Quels sont les meilleurs produits ?",
        "Quelles sont les promotions ?",
        "Quelles boutiques sont disponibles ?"
    ]

    for q in test_questions:
        response = enricher.chatbot_response(q, df)
        print(f"\n  Q: {q}")
        print(f"  R: {response}")

    print("\n" + "=" * 60)
    print("  ETAPE 5 TERMINEE")
    print("=" * 60)


if __name__ == "__main__":
    # Choisir quoi lancer :

    # Etape 3 (deja fait normalement)
    # run_ml_pipeline()

    # Etape 5 : LLM
    run_llm_enrichment()
