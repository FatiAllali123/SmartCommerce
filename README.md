# Smart eCommerce Intelligence

Projet Data Mining — Analyse intelligente de produits e-commerce avec Machine Learning, Kubeflow Pipeline, LLM et Dashboard BI.

## Architecture du projet


projet_ecommerce/
├── scraping/ # Agents A2A (Shopify + WooCommerce)
│ ├── agents.py
│ ├── orchestrator.py
│ └── config.py
├── ml/ # Modeles Machine Learning
│ ├── preprocessing.py
│ ├── models.py
│ └── evaluation.py
├── llm/ # Module LLM (Groq / Llama 3.1)
│ └── enrichment.py
├── mcp/ # Architecture MCP
│ └── mcp_simulation.py
├── pipeline/ # Pipeline Kubeflow
│ ├── kubeflow_pipeline.py
│ ├── pipeline.yaml
│ └── Dockerfile
├── dashboard/ # Dashboard BI (Streamlit)
│ └── app.py
├── data/
│ ├── raw/
│ └── processed/
├── outputs/
│ ├── models/
│ ├── figures/
│ ├── top_k_products.csv
│ ├── model_comparison.csv
│ ├── association_rules.csv
│ └── anomalies.csv
├── .github/workflows/
├── Dockerfile
├── requirements.txt
└── README.md


## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/VOTRE_USERNAME/projet_ecommerce.git
cd projet_ecommerce
2. Creer l'environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
3. Installer les dependances
pip install -r requirements.txt
4. Configurer la cle API Groq (optionnel)

Creez un fichier .env a la racine :

GROQ_API_KEY=votre_cle_ici

Sans cle API, le module LLM fonctionne en mode regles.

Lancement
Scraping (donnees deja dans data/raw/)
python -m scraping.orchestrator
Pipeline Kubeflow (Preprocessing + ML + Evaluation + Top-K)
python pipeline/kubeflow_pipeline.py

Execute les 4 etapes :

Preprocessing : nettoyage, feature engineering, score composite
Training : Random Forest, XGBoost, KMeans
Evaluation : accuracy, F1, silhouette score
Top-K Selection : top 20 produits
Modeles ML complets + LLM
python main.py

Lance tous les modeles ML et le module LLM (resumes, rapports, chatbot).

Dashboard BI
streamlit run dashboard/app.py

7 pages : Vue d'ensemble, Top-K, Clusters, Anomalies, Regles d'association, Comparaison ML, Chatbot IA.

Docker
docker build -t ecommerce-dashboard .
docker run -p 8501:8501 ecommerce-dashboard
Modeles ML utilises
Modele	Type	Usage
Random Forest	Supervise	Predire le succes d'un produit
XGBoost	Supervise	Prediction avancee
KMeans	Non-supervise	Segmentation des produits
DBSCAN	Non-supervise	Detection d'anomalies
Clustering hierarchique	Non-supervise	Dendrogramme de segmentation
PCA	Visualisation	Reduction 2D pour affichage
FP-Growth	Association	Regles d'association
Score composite
Critere	Poids	Justification
Disponibilite	25%	Un produit indisponible ne peut pas reussir
Prix	20%	Prix dans la fourchette optimale
Variantes	15%	Plus de choix = plus de conversions
Images	15%	Visuels = confiance client
Description	15%	Description riche = meilleur SEO
Remise	10%	Promotion = attractivite

Variable cible : produit_succes = 1 si top 20% par score.

Module LLM
Mode LLM : Groq API avec Llama 3.1 8B (Chain of Thought)
Mode regles : rapports generes sans API
Fonctionnalites : resumes, analyse concurrentielle, tendances, strategies marketing, chatbot, profil client, prompts de scraping, nettoyage LLM
Architecture MCP
MCPHost (Dashboard) : orchestre les requetes
MCPClient (LLM) : envoie les requetes avec permissions
MCPServer (Scraping) : fournit les donnees
Systeme de permissions + journal d'audit
Donnees
Source : 4 boutiques Shopify + 2 WooCommerce
Volume : 4769 produits scrapes, 4636 apres nettoyage
Variables : 43 colonnes (20 initiales + 23 derivees)
Boutiques : Gymshark, Allbirds, Brooklinen, Pura Vida Bracelets
Limitations connues
Donnees de popularite (rating, nb_reviews)

Les APIs Shopify JSON (/products.json) ne fournissent pas les notes et nombre d'avis.
Ces donnees necessitent l'acces a l'API Admin Shopify ou un scraping HTML avec Selenium/Playwright.

Le scoring utilise des proxies : disponibilite, prix, remise, description, variantes et images.

Scraping JavaScript dynamique

Les boutiques Shopify exposees utilisent une API JSON (/products.json), rendant Selenium/Playwright inutile pour ces sites.

Pour des sites sans API JSON, Playwright serait necessaire.

WooCommerce

L'agent WooCommerce est implemente avec double fallback (API REST + HTML) mais les boutiques testees ont repondu avec des erreurs 401/403.

L'agent reste fonctionnel et testable avec d'autres boutiques.

Technologies
Python 3.10+
ML : scikit-learn, xgboost, mlxtend
LLM : LangChain, Groq (Llama 3.1)
Dashboard : Streamlit, Plotly
Pipeline : Kubeflow Pipelines SDK
Scraping : requests, BeautifulSoup
Containerisation : Docker
CI/CD : GitHub Actions
