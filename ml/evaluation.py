"""
=============================================================
EVALUATION ET VISUALISATION DES MODELES
=============================================================

Ce fichier genere les graphiques et tableaux d'evaluation
que le prof veut voir dans le rapport et le dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Genere les visualisations pour l'evaluation des modeles."""

    def __init__(self, output_dir="outputs/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_test, y_pred, title="Matrice de confusion"):
        """
        Affiche la matrice de confusion.

        La matrice de confusion montre :
        - Combien de "succes" ont ete correctement predits (VP)
        - Combien de "non-succes" ont ete correctement predits (VN)
        - Combien d'erreurs il y a eu (FP, FN)
        """
        cm = np.array(y_test) if hasattr(y_test, '__len__') else y_test
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-succes', 'Succes'],
            yticklabels=['Non-succes', 'Succes'],
            ax=ax
        )
        ax.set_xlabel('Predit')
        ax.set_ylabel('Reel')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"confusion_matrix_{title.lower().replace(' ', '_')}.png"))
        plt.close()
        logger.info(f"  Figure sauvegardee : confusion_matrix_{title.lower().replace(' ', '_')}.png")

    def plot_feature_importance(self, importance_df, title="Importance des variables"):
        """Affiche l'importance des variables sous forme de barres."""
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = importance_df.head(15)
        ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"feature_importance_{title.lower().replace(' ', '_')}.png"))
        plt.close()
        logger.info(f"  Figure sauvegardee : feature_importance.png")

    def plot_clusters_2d(self, X_pca, cluster_labels, title="Clusters PCA"):
        """Affiche les clusters dans un espace 2D (apres PCA)."""
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=cluster_labels, cmap='viridis',
            alpha=0.6, s=20, edgecolors='none'
        )
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel('Composante principale 1')
        ax.set_ylabel('Composante principale 2')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "clusters_pca.png"))
        plt.close()
        logger.info(f"  Figure sauvegardee : clusters_pca.png")

    def plot_silhouette_scores(self, k_range, scores, optimal_k):
        """Affiche le silhouette score pour differentes valeurs de K."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_range, scores, 'bo-', linewidth=2, markersize=8)
        ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'K optimal = {optimal_k}')
        ax.set_xlabel('Nombre de clusters (K)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Recherche du K optimal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "silhouette_scores.png"))
        plt.close()
        logger.info(f"  Figure sauvegardee : silhouette_scores.png")

    def plot_price_distribution(self, df):
        """Distribution des prix par boutique."""
        fig, ax = plt.subplots(figsize=(12, 6))
        stores = df['store_name'].unique()
        for store in stores:
            store_data = df[df['store_name'] == store]
            ax.hist(store_data['price'], bins=30, alpha=0.5, label=store)
        ax.set_xlabel('Prix')
        ax.set_ylabel('Nombre de produits')
        ax.set_title('Distribution des prix par boutique')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "price_distribution.png"))
        plt.close()
        logger.info(f"  Figure sauvegardee : price_distribution.png")

    def generate_summary_table(self, rf_results, xgb_results):
        """Genere un tableau comparatif des modeles."""
        comparison = pd.DataFrame({
            'Metrique': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'CV F1 (mean)'],
            'Random Forest': [
                rf_results.get('accuracy', 0),
                rf_results.get('precision', 0),
                rf_results.get('recall', 0),
                rf_results.get('f1', 0),
                rf_results.get('cv_f1_mean', 0)
            ],
            'XGBoost': [
                xgb_results.get('accuracy', 0),
                xgb_results.get('precision', 0),
                xgb_results.get('recall', 0),
                xgb_results.get('f1', 0),
                xgb_results.get('cv_f1_mean', 0)
            ]
        })

        logger.info("\n--- COMPARAISON DES MODELES ---")
        logger.info(f"\n{comparison.to_string(index=False)}")

        return comparison
