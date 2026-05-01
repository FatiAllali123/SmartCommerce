"""
=============================================================
MODELES ML POUR LE PROJET ECOMMERCE
=============================================================

Ce fichier contient tous les modeles de Data Mining :
1. RandomForest      (supervise) - predire le succes d'un produit
2. XGBoost           (supervise) - predire le succes (plus performant)
3. KMeans            (non-supervise) - segmenter les produits
4. DBSCAN            (non-supervise) - detecter les anomalies
5. Regles d'association - decouvrir les associations produits

Pourquoi ces modeles ?
→ Le prof les demande tous dans le projet.
→ Chaque modele repond a une question business differente.
"""

import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    silhouette_score
)
from xgboost import XGBClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# ==============================================================
# 1. RANDOM FOREST (SUPERVISE)
# ==============================================================
class RandomForestModel:
    """
    Random Forest : modele de classification supervisee.

    Comment ca marche (simple) ?
    → Imaginez que vous demandez a 100 personnes :
       "Ce produit va-t-il reussir ?"
    → Chaque personne regarde quelques criteres (prix, note, etc.)
    → La reponse finale = la majorite des 100 avis
    → C'est ca une "foret" : beaucoup d'"arbres de decision" votes

    Pourquoi on l'utilise ?
    → C'est un bon modele de base (baseline)
    → Il est robuste et facile a interpreter
    → Il montre quelles variables sont les plus importantes
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.results = {}

    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare les donnees pour l'entrainement.

        On separe les donnees en deux parties :
        - 80% pour entrainer le modele (train)
        - 20% pour tester le modele (test)

        Pourquoi ?
        → On entraine sur des donnees connues
        → On teste sur des donnees que le modele n'a jamais vues
        → Ca permet de savoir si le modele est vraiment bon
        """
        # Variables explicatives (ce que le modele utilise pour predire)
        feature_cols = [
            'price', 'discount_pct', 'variants_count', 'images_count',
            'is_available', 'inventory_quantity', 'tags_count',
            'word_count_description', 'description_length',
            'title_length', 'has_discount', 'has_images',
            'has_variants', 'product_type_encoded', 'store_encoded',
            'country_encoded', 'product_age_days'
        ]

        # On garde seulement les colonnes qui existent
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].fillna(0)
        y = df['produit_succes']

        self.feature_names = feature_cols

        # Separation train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Donnees preparees :")
        logger.info(f"  Train : {len(X_train)} produits")
        logger.info(f"  Test  : {len(X_test)} produits")
        logger.info(f"  Variables : {len(feature_cols)}")
        logger.info(f"  Variable cible : produit_succes (0/1)")

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Entraine le modele Random Forest.

        Parametres du modele :
        - n_estimators=100 : 100 arbres de decision
        - max_depth=10 : chaque arbre a 10 niveaux max
        - random_state=42 : pour la reproductibilite
        """
        logger.info("\n--- RANDOM FOREST ---")
        logger.info("Entrainement en cours...")

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)
        logger.info("Entrainement termine")

        # Validation croisee (5-fold)
        # → On decoupe les donnees en 5 morceaux
        # → On entraine sur 4 et teste sur 1, 5 fois
        # → Ca donne une estimation plus fiable
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        self.results['cv_f1_mean'] = cv_scores.mean()
        self.results['cv_f1_std'] = cv_scores.std()
        logger.info(f"Validation croisee F1 : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.model

    def evaluate(self, X_test, y_test):
        """
        Evalue le modele sur les donnees de test.

        Metriques :
        - Accuracy : % de bonnes reponses globales
        - Precision : parmi les produits predits "succes", combien le sont vraiment
        - Recall : parmi les vrais succes, combien le modele a trouves
        - F1-score : moyenne de precision et recall
        - Matrice de confusion : tableau detaille des predictions
        """
        y_pred = self.model.predict(X_test)

        self.results['accuracy'] = accuracy_score(y_test, y_pred)
        self.results['precision'] = precision_score(y_test, y_pred)
        self.results['recall'] = recall_score(y_test, y_pred)
        self.results['f1'] = f1_score(y_test, y_pred)
        self.results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        self.results['classification_report'] = classification_report(y_test, y_pred)
        self.results['y_test'] = y_test
        self.results['y_pred'] = y_pred

        logger.info(f"\nResultats Random Forest :")
        logger.info(f"  Accuracy  : {self.results['accuracy']:.4f}")
        logger.info(f"  Precision : {self.results['precision']:.4f}")
        logger.info(f"  Recall    : {self.results['recall']:.4f}")
        logger.info(f"  F1-score  : {self.results['f1']:.4f}")
        logger.info(f"\n{self.results['classification_report']}")

        return self.results

    def get_feature_importance(self):
        """
        Affiche l'importance de chaque variable.

        Ca nous dit quelles variables le modele utilise le plus
        pour decider si un produit va reussir.

        Exemple : si "price" a une importance de 0.25,
        ca veut dire que le prix compte pour 25% de la decision.
        """
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nImportance des variables (Random Forest) :")
        for _, row in importance.iterrows():
            bar = "█" * int(row['importance'] * 50)
            logger.info(f"  {row['feature']:30s} : {row['importance']:.4f} {bar}")

        return importance


# ==============================================================
# 2. XGBOOST (SUPERVISE)
# ==============================================================
class XGBoostModel:
    """
    XGBoost : modele de classification avancee.

    Comment ca marche (simple) ?
    → Comme Random Forest, mais chaque nouvel arbre
       corrige les erreurs du precedent.
    → C'est comme une equipe qui s'ameliorer a chaque match.

    Pourquoi on l'utilise ?
    → Generalement plus precis que Random Forest
    → Tres utilise en competition Kaggle
    → Le prof le demande explicitement
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.results = {}

    def train(self, X_train, y_train, feature_names=None):
        """Entraine le modele XGBoost."""
        logger.info("\n--- XGBOOST ---")
        logger.info("Entrainement en cours...")

        self.feature_names = feature_names

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

        self.model.fit(X_train, y_train)
        logger.info("Entrainement termine")

        # Validation croisee
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        self.results['cv_f1_mean'] = cv_scores.mean()
        self.results['cv_f1_std'] = cv_scores.std()
        logger.info(f"Validation croisee F1 : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return self.model

    def evaluate(self, X_test, y_test):
        """Evalue le modele XGBoost."""
        y_pred = self.model.predict(X_test)

        self.results['accuracy'] = accuracy_score(y_test, y_pred)
        self.results['precision'] = precision_score(y_test, y_pred)
        self.results['recall'] = recall_score(y_test, y_pred)
        self.results['f1'] = f1_score(y_test, y_pred)
        self.results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        self.results['classification_report'] = classification_report(y_test, y_pred)
        self.results['y_test'] = y_test
        self.results['y_pred'] = y_pred

        logger.info(f"\nResultats XGBoost :")
        logger.info(f"  Accuracy  : {self.results['accuracy']:.4f}")
        logger.info(f"  Precision : {self.results['precision']:.4f}")
        logger.info(f"  Recall    : {self.results['recall']:.4f}")
        logger.info(f"  F1-score  : {self.results['f1']:.4f}")
        logger.info(f"\n{self.results['classification_report']}")

        return self.results

    def get_feature_importance(self):
        """Affiche l'importance des variables pour XGBoost."""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nImportance des variables (XGBoost) :")
        for _, row in importance.iterrows():
            bar = "█" * int(row['importance'] * 50)
            logger.info(f"  {row['feature']:30s} : {row['importance']:.4f} {bar}")

        return importance


# ==============================================================
# 3. KMEANS (NON-SUPERVISE)
# ==============================================================
class KMeansModel:
    """
    KMeans : algorithme de clustering.

    Comment ca marche (simple) ?
    → On dit a l'ordinateur : "Regroupe ces produits en K groupes"
    → L'ordinateur trouve les groupes tout seul
    → Chaque groupe contient des produits similaires

    Pourquoi on l'utilise ?
    → Pour segmenter les produits (premium, discount, populaires)
    → Ca aide a comprendre la structure du marche

    Nombre de clusters : on teste 2, 3, 4, 5 et on garde le meilleur
    (on utilise le silhouette score pour choisir)
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.results = {}

    def find_optimal_k(self, X_scaled, max_k=8):
        """
        Trouve le nombre optimal de clusters.

        On teste differentes valeurs de K (2, 3, 4, ..., 8)
        et on garde celle avec le meilleur silhouette score.

        Silhouette score : mesure a quel point les groupes sont separes
        → Proche de 1 = tres bien separe
        → Proche de 0 = les groupes se melangent
        → Negatif = les groupes sont mal definis
        """
        logger.info("\n--- KMEANS : Recherche du K optimal ---")

        silhouette_scores = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
            logger.info(f"  K={k} : silhouette = {score:.4f}")

        # Le meilleur K
        best_k = list(K_range)[np.argmax(silhouette_scores)]
        best_score = max(silhouette_scores)
        logger.info(f"\n  Meilleur K = {best_k} (silhouette = {best_score:.4f})")

        return best_k, silhouette_scores

    def train(self, X_scaled, n_clusters=3):
        """Entraine KMeans avec le nombre de clusters choisi."""
        logger.info(f"\n--- KMEANS avec K={n_clusters} ---")

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )

        labels = self.model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)

        self.results['labels'] = labels
        self.results['silhouette'] = score
        self.results['n_clusters'] = n_clusters
        self.results['inertia'] = self.model.inertia_

        logger.info(f"  Clusters : {n_clusters}")
        logger.info(f"  Silhouette score : {score:.4f}")

        # Taille de chaque cluster
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            pct = count / len(labels) * 100
            logger.info(f"  Cluster {cluster_id} : {count} produits ({pct:.1f}%)")

        return labels

    def interpret_clusters(self, df, feature_cols):
        """
        Interprete les clusters en affichant les caracteristiques moyennes.

        Ca nous dit ce qui definit chaque groupe :
        - Cluster 0 : peut-etre les produits premium (prix eleve)
        - Cluster 1 : peut-etre les produits discount (remises fortes)
        - Cluster 2 : peut-etre les produits populaires (beaucoup d'avis)
        """
        df_temp = df.copy()
        df_temp['cluster'] = self.results['labels']

        logger.info("\nInterpretation des clusters :")

        # Moyennes par cluster
        cluster_means = df_temp.groupby('cluster')[feature_cols].mean()
        logger.info(f"\n{cluster_means.round(2).to_string()}")

        # Caracteristiques principales par cluster
        for cluster_id in sorted(df_temp['cluster'].unique()):
            cluster_data = df_temp[df_temp['cluster'] == cluster_id]
            logger.info(f"\n  Cluster {cluster_id} ({len(cluster_data)} produits) :")
            logger.info(f"    Prix moyen    : {cluster_data['price'].mean():.2f}")
            logger.info(f"    Remise moy    : {cluster_data['discount_pct'].mean():.1f}%")
            logger.info(f"    Taux succes   : {cluster_data['produit_succes'].mean()*100:.1f}%")
            logger.info(f"    Boutique top  : {cluster_data['store_name'].mode().values[0]}")

        return cluster_means


# ==============================================================
# 4. DBSCAN (NON-SUPERVISE)
# ==============================================================
class DBSCANModel:
    """
    DBSCAN : detection d'anomalies.

    Comment ca marche (simple) ?
    → DBSCAN regroupe les produits proches les uns des autres
    → Les produits isoles (loin de tout groupe) = anomalies
    → Anomalie = produit atypique (prix bizarre, description etrange, etc.)

    Pourquoi on l'utilise ?
    → Pour detecter les produits bizarres dans les donnees
    → Exemple : un produit a 0.50€ dans une boutique premium
    """

    def __init__(self):
        self.labels = None
        self.results = {}

    def run(self, X_scaled):
        """Lance DBSCAN."""
        logger.info("\n--- DBSCAN ---")

        dbscan = DBSCAN(eps=1.5, min_samples=10)
        self.labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_anomalies = list(self.labels).count(-1)

        self.results['labels'] = self.labels
        self.results['n_clusters'] = n_clusters
        self.results['n_anomalies'] = n_anomalies

        logger.info(f"  Clusters trouves : {n_clusters}")
        logger.info(f"  Anomalies detectees : {n_anomalies} ({n_anomalies/len(self.labels)*100:.1f}%)")

        if n_clusters > 0:
            non_anomaly = self.labels != -1
            if non_anomaly.sum() > 1:
                score = silhouette_score(X_scaled[non_anomaly], self.labels[non_anomaly])
                logger.info(f"  Silhouette (sans anomalies) : {score:.4f}")

        return self.labels

    def analyze_anomalies(self, df):
        """Analyse les anomalies detectees."""
        df_temp = df.copy()
        df_temp['dbscan_label'] = self.labels
        anomalies = df_temp[df_temp['dbscan_label'] == -1]

        if len(anomalies) > 0:
            logger.info(f"\nExemples d'anomalies :")
            for _, row in anomalies.head(10).iterrows():
                logger.info(f"  {row['title'][:50]:50s} | Prix: {row['price']:>8.2f} | {row['store_name']}")

        return anomalies

# ==============================================================
# 5b. CLUSTERING HIERARCHIQUE (NON-SUPERVISE)
# ==============================================================
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class HierarchicalClusteringModel:
    """
    Clustering hierarchique : regroupement agglomeratif.

    Comment ca marche (simple) ?
    → On commence avec chaque produit comme un cluster separe
    → A chaque etape, on fusionne les 2 clusters les plus proches
    → On obtient un arbre (dendrogramme) qui montre les fusions

    Difference avec KMeans :
    → KMeans : on decide du nombre de clusters a l'avance
    → Hierarchique : on voit la structure complete, puis on decide

    Pourquoi on l'utilise ?
    → Le prof le demande explicitement dans l'enonce
    → Le dendrogramme est un outil visuel puissant
    """

    def __init__(self):
        self.model = None
        self.results = {}

    def train(self, X_scaled, n_clusters=3):
        """Entraine le clustering hierarchique."""
        logger.info("\n--- CLUSTERING HIERARCHIQUE ---")

        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )

        labels = self.model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        self.results['labels'] = labels
        self.results['silhouette'] = score
        self.results['n_clusters'] = n_clusters

        logger.info(f"  Clusters : {n_clusters}")
        logger.info(f"  Silhouette score : {score:.4f}")

        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            pct = count / len(labels) * 100
            logger.info(f"  Cluster {cluster_id} : {count} produits ({pct:.1f}%)")

        return labels

    def plot_dendrogram(self, X_scaled, output_path="outputs/figures/dendrogram.png"):
        """
        Genere le dendrogramme.

        Le dendrogramme montre visuellement comment les clusters
        se forment a chaque etape de fusion.
        """
        logger.info("  Generation du dendrogramme...")

        # On prend un echantillon pour que le dendrogramme soit lisible
        n_sample = min(200, len(X_scaled))
        indices = np.random.choice(len(X_scaled), n_sample, replace=False)
        X_sample = X_scaled[indices]

        # Calcul de la matrice de liaison
        Z = linkage(X_sample, method='ward')

        # Dessiner le dendrogramme
        fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(Z, ax=ax, truncate_mode='level', p=5,
                  leaf_font_size=8, color_threshold=0.7*max(Z[:,2]))
        ax.set_title('Dendrogramme - Clustering Hierarchique')
        ax.set_xlabel('Produits')
        ax.set_ylabel('Distance')
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"  Dendrogramme sauvegarde : {output_path}")

        return Z

# ==============================================================
# 5. PCA (VISUALISATION)
# ==============================================================
class PCAModel:
    """
    PCA : reduction de dimension pour visualisation.

    Comment ca marche (simple) ?
    → On a 17 variables (prix, remise, etc.)
    → On ne peut pas visualiser 17 dimensions
    → PCA compresse tout en 2 dimensions
    → On peut dessiner un graphique 2D

    Pourquoi on l'utilise ?
    → Pour visualiser les clusters et les anomalies
    → Pour montrer au prof qu'on sait faire de la visualisation
    """

    def __init__(self):
        self.pca = None
        self.X_pca = None
        self.results = {}

    def fit_transform(self, X_scaled, n_components=2):
        """Reduit les dimensions a 2 pour la visualisation."""
        logger.info("\n--- PCA (Visualisation 2D) ---")

        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(X_scaled)

        explained_var = self.pca.explained_variance_ratio_
        logger.info(f"  Variance expliquee :")
        logger.info(f"    Composante 1 : {explained_var[0]*100:.1f}%")
        logger.info(f"    Composante 2 : {explained_var[1]*100:.1f}%")
        logger.info(f"    Total         : {sum(explained_var)*100:.1f}%")

        self.results['explained_variance'] = explained_var
        self.results['X_pca'] = self.X_pca

        return self.X_pca





# ==============================================================
# 6. REGLES D'ASSOCIATION
# ==============================================================
from mlxtend.frequent_patterns import fpgrowth

class AssociationRulesMiner:
    """
    Regles d'association : decouvrir quels elements vont ensemble.
    On utilise FP-Growth (rapide) avec des seuils eleves
    pour eviter l'explosion combinatoire.
    """

    def __init__(self):
        self.results = {}

    def run(self, df, min_support=0.05, min_confidence=0.3):
        """
        Lance la recherche de regles d'association.
        """
        logger.info("\n--- REGLES D'ASSOCIATION ---")

        all_rules = []

        # APPROCHE 1 : Tags produits
        logger.info("\n  [Approche 1] Association par tags produits")
        rules_tags = self._rules_from_tags(df)
        if rules_tags is not None and len(rules_tags) > 0:
            rules_tags['source'] = 'tags'
            all_rules.append(rules_tags)

        # APPROCHE 2 : Categories par boutique
        logger.info("\n  [Approche 2] Association par categories/boutique")
        rules_store = self._rules_from_store_categories(df)
        if rules_store is not None and len(rules_store) > 0:
            rules_store['source'] = 'store_categories'
            all_rules.append(rules_store)

        # APPROCHE 3 : Attributs produits
        logger.info("\n  [Approche 3] Association par attributs produits")
        rules_attrs = self._rules_from_attributes(df)
        if rules_attrs is not None and len(rules_attrs) > 0:
            rules_attrs['source'] = 'attributes'
            all_rules.append(rules_attrs)

        # Combiner
        if all_rules:
            combined = pd.concat(all_rules, ignore_index=True)
            combined = combined.sort_values('lift', ascending=False)
            self.results['rules'] = combined

            logger.info(f"\n  TOTAL regles trouvees : {len(combined)}")
            logger.info(f"\n  Top 10 regles d'association :")
            for i, (_, rule) in enumerate(combined.head(10).iterrows(), 1):
                antecedent = ', '.join(list(rule['antecedents']))
                consequent = ', '.join(list(rule['consequents']))
                logger.info(f"  {i}. {{{antecedent}}} → {{{consequent}}}")
                logger.info(f"     Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f} [{rule.get('source', '')}]")

            return combined
        else:
            logger.warning("  Aucune regle trouvee.")
            return None

    def _rules_from_tags(self, df):
        """
        Regles a partir des tags.
        On garde seulement 8 tags max et un support eleve.
        """
        df_tags = df[df['tags'].notna()].copy()
        df_tags = df_tags[df_tags['tags'].str.strip() != '']

        if len(df_tags) < 50:
            return None

        # Tags a ignorer
        ignore_tags = {
            'all-products', 'shoprunner', 'build-your-wishlist',
            'sizeguide:top', 'sizeguide:bottom', 'sizeguide:outerwear',
            'aw25', 'ss26', 'aw24', 'ss25', 'season:ss26', 'season:aw25',
            'spanish-translated', 'french-translated',
            'over-50-off', 'over-70-off', 'over-30-off', 'over-10-off',
            'new-arrivals', 'best-sellers', 'sale',
            'homepage', 'nav', 'hidden', 'draft',
        }

        # Transactions
        transactions = []
        for _, row in df_tags.iterrows():
            tags = [t.strip().lower() for t in str(row['tags']).split(',')]
            tags = [t for t in tags if t and len(t) > 2 and t not in ignore_tags]
            if len(tags) >= 2:
                transactions.append(tags)

        if len(transactions) < 50:
            return None

        # Compter et garder SEULEMENT les 8 tags les plus frequents
        from collections import Counter
        tag_counts = Counter()
        for t in transactions:
            tag_counts.update(t)

        n_trans = len(transactions)
        top_tags = set()
        for tag, count in tag_counts.most_common(100):
            freq = count / n_trans
            if 0.10 <= freq <= 0.70:
                top_tags.add(tag)
            if len(top_tags) >= 8:
                break

        logger.info(f"  Tags retenus ({len(top_tags)}) : {sorted(top_tags)}")

        if len(top_tags) < 3:
            return None

        # Filtrer transactions
        transactions_filtered = []
        for t in transactions:
            filtered = [tag for tag in t if tag in top_tags]
            if len(filtered) >= 2:
                transactions_filtered.append(filtered)

        logger.info(f"  Transactions filtrees : {len(transactions_filtered)}")

        if len(transactions_filtered) < 50:
            return None

        # Encodage
        te = TransactionEncoder()
        te_array = te.fit(transactions_filtered).transform(transactions_filtered)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        # FP-Growth avec support eleve et max_len=3
        # max_len=3 empeche les ensembles de plus de 3 items
        frequent_itemsets = fpgrowth(
            df_encoded, min_support=0.15, use_colnames=True, max_len=3
        )
        logger.info(f"  Ensembles frequents : {len(frequent_itemsets)}")

        if len(frequent_itemsets) == 0:
            return None

        # Regles
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        logger.info(f"  Regles tags : {len(rules)}")

        return rules if len(rules) > 0 else None

    def _rules_from_store_categories(self, df):
        """
        Regles a partir des categories par boutique.
        """
        df_temp = df.copy()
        df_temp['type_simple'] = df_temp['product_type'].apply(
            lambda x: str(x).split('>')[0].strip() if pd.notna(x) and str(x).strip() else None
        )
        df_temp = df_temp[df_temp['type_simple'].notna()]
        df_temp = df_temp[df_temp['type_simple'] != '']
        df_temp = df_temp[df_temp['type_simple'] != 'nan']

        transactions = []
        for store in df_temp['store_name'].unique():
            store_products = df_temp[df_temp['store_name'] == store]
            types = store_products['type_simple'].unique().tolist()
            types = [t for t in types if t and t.strip()]
            if len(types) >= 2:
                transactions.append(types)

        logger.info(f"  Transactions (boutiques) : {len(transactions)}")

        if len(transactions) < 2:
            return None

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        frequent_itemsets = fpgrowth(
            df_encoded, min_support=0.3, use_colnames=True, max_len=3
        )
        logger.info(f"  Ensembles frequents : {len(frequent_itemsets)}")

        if len(frequent_itemsets) == 0:
            return None

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        logger.info(f"  Regles boutiques : {len(rules)}")

        return rules if len(rules) > 0 else None

    def _rules_from_attributes(self, df):
        """
        Regles a partir des attributs produits.
        """
        transactions = []
        for _, row in df.iterrows():
            items = []

            if pd.notna(row.get('store_name')):
                items.append(f"store:{row['store_name']}")

            if pd.notna(row.get('product_type')):
                cat = str(row['product_type']).split('>')[0].strip()
                if cat and cat != 'nan':
                    items.append(f"cat:{cat}")

            if pd.notna(row.get('price_category')):
                items.append(f"price:{row['price_category']}")

            if row.get('is_available') == 1:
                items.append("available:yes")
            else:
                items.append("available:no")

            if row.get('has_discount') == 1:
                items.append("discount:yes")
            else:
                items.append("discount:no")

            if len(items) >= 3:
                transactions.append(items)

        logger.info(f"  Transactions (attributs) : {len(transactions)}")

        if len(transactions) < 100:
            return None

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        frequent_itemsets = fpgrowth(
            df_encoded, min_support=0.10, use_colnames=True, max_len=3
        )
        logger.info(f"  Ensembles frequents : {len(frequent_itemsets)}")

        if len(frequent_itemsets) == 0:
            return None

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        logger.info(f"  Regles attributs : {len(rules)}")

        return rules if len(rules) > 0 else None
