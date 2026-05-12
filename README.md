# 🛒 Smart eCommerce Intelligence

> Projet Data Mining — Analyse intelligente de produits e-commerce avec Machine Learning, Kubeflow Pipeline, LLM et Dashboard BI.

---

## 📁 Architecture du projet

```bash
projet_ecommerce/
├── scraping/               # Agents A2A (Shopify + WooCommerce)
│   ├── agents.py
│   ├── orchestrator.py
│   └── config.py
├── ml/                     # Modèles Machine Learning
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── llm/                    # Module LLM (Groq / Llama 3.1)
│   └── enrichment.py
├── mcp/                    # Architecture MCP
│   └── mcp_simulation.py
├── pipeline/               # Pipeline Kubeflow
│   ├── kubeflow_pipeline.py
│   ├── pipeline.yaml
│   └── Dockerfile
├── dashboard/              # Dashboard BI (Streamlit)
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── models/
│   ├── figures/
│   ├── top_k_products.csv
│   ├── model_comparison.csv
│   ├── association_rules.csv
│   └── anomalies.csv
├── .github/workflows/
├── Dockerfile
├── requirements.txt
└── README.md
```
---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/VOTRE_USERNAME/projet_ecommerce.git
cd projet_ecommerce
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API Groq *(optionnel)*

Créez un fichier `.env` à la racine :

```env
GROQ_API_KEY=votre_cle_ici
```

> Sans clé API, le module LLM fonctionne en **mode règles**.

---

## 🚀 Lancement

### Scraping *(données déjà dans `data/raw/`)*

```bash
python -m scraping.orchestrator
```

### Pipeline Kubeflow — Preprocessing + ML + Évaluation + Top-K

```bash
python pipeline/kubeflow_pipeline.py
```

Exécute les 4 étapes :

1. **Preprocessing** — nettoyage, feature engineering, score composite
2. **Training** — Random Forest, XGBoost, KMeans
3. **Évaluation** — accuracy, F1, silhouette score
4. **Top-K Selection** — top 20 produits

### Modèles ML complets + LLM

```bash
python main.py
```

Lance tous les modèles ML et le module LLM (résumés, rapports, chatbot).

### Dashboard BI

```bash
streamlit run dashboard/app.py
```

7 pages : Vue d'ensemble · Top-K · Clusters · Anomalies · Règles d'association · Comparaison ML · Chatbot IA

### Docker

```bash
docker build -t ecommerce-dashboard .
docker run -p 8501:8501 ecommerce-dashboard
```

---

## 🤖 Modèles ML utilisés

| Modèle | Type | Usage |
|---|---|---|
| Random Forest | Supervisé | Prédire le succès d'un produit |
| XGBoost | Supervisé | Prédiction avancée |
| KMeans | Non-supervisé | Segmentation des produits |
| DBSCAN | Non-supervisé | Détection d'anomalies |
| Clustering hiérarchique | Non-supervisé | Dendrogramme de segmentation |
| PCA | Visualisation | Réduction 2D pour affichage |
| FP-Growth | Association | Règles d'association |

---

## 📊 Score composite

| Critère | Poids | Justification |
|---|---|---|
| Disponibilité | 25% | Un produit indisponible ne peut pas réussir |
| Prix | 20% | Prix dans la fourchette optimale |
| Variantes | 15% | Plus de choix = plus de conversions |
| Images | 15% | Visuels = confiance client |
| Description | 15% | Description riche = meilleur SEO |
| Remise | 10% | Promotion = attractivité |

> **Variable cible :** `produit_succes = 1` si top 20% par score.

---

## 🧠 Module LLM

- **Mode LLM** — Groq API avec Llama 3.1 8B (Chain of Thought)
- **Mode règles** — rapports générés sans API
- **Fonctionnalités** — résumés, analyse concurrentielle, tendances, stratégies marketing, chatbot, profil client, prompts de scraping, nettoyage LLM

---

## 🏗️ Architecture MCP

| Composant | Rôle |
|---|---|
| MCPHost (Dashboard) | Orchestre les requêtes |
| MCPClient (LLM) | Envoie les requêtes avec permissions |
| MCPServer (Scraping) | Fournit les données |

Système de permissions + journal d'audit.

---

## 📦 Données

| Attribut | Détail |
|---|---|
| Source | 4 boutiques Shopify + 2 WooCommerce |
| Volume | 4 769 produits scrapés, 4 636 après nettoyage |
| Variables | 43 colonnes (20 initiales + 23 dérivées) |
| Boutiques | Gymshark, Allbirds, Brooklinen, Pura Vida Bracelets |

---

## ⚠️ Limitations connues

### Données de popularité (rating, nb_reviews)

Les APIs Shopify JSON (`/products.json`) ne fournissent pas les notes et nombre d'avis. Ces données nécessitent l'accès à l'API Admin Shopify ou un scraping HTML avec Selenium/Playwright.

Le scoring utilise des proxies : disponibilité, prix, remise, description, variantes et images.

### Scraping JavaScript dynamique

Les boutiques Shopify exposées utilisent une API JSON (`/products.json`), rendant Selenium/Playwright inutile pour ces sites. Pour des sites sans API JSON, Playwright serait nécessaire.

### WooCommerce

L'agent WooCommerce est implémenté avec double fallback (API REST + HTML) mais les boutiques testées ont répondu avec des erreurs `401`/`403`. L'agent reste fonctionnel et testable avec d'autres boutiques.

---

## 🛠️ Technologies

| Catégorie | Outils |
|---|---|
| Langage | Python 3.10+ |
| Machine Learning | scikit-learn, xgboost, mlxtend |
| LLM | LangChain, Groq (Llama 3.1) |
| Dashboard | Streamlit, Plotly |
| Pipeline | Kubeflow Pipelines SDK |
| Scraping | requests, BeautifulSoup |
| Conteneurisation | Docker |
| CI/CD | GitHub Actions |
