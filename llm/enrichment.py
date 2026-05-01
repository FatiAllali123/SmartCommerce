"""
=============================================================
MODULE LLM - ENRICHISSEMENT ET SYNTHESE
=============================================================

Ce module utilise un LLM pour :
1. Resumer les descriptions produits
2. Generer des rapports concurrentiels
3. Proposer des strategies marketing
4. Creer des profils clients
5. Alimenter un chatbot dans le dashboard

Fonctionne avec OU sans cle API OpenAI :
- Avec cle API : utilise GPT-4o-mini (meilleur resultat)
- Sans cle API : genere des rapports bases sur les donnees (regles)
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables du fichier .env
load_dotenv()

logger = logging.getLogger(__name__)


class LLMEnricher:
    """
    Classe principale pour l'enrichissement par LLM.

    Deux modes :
    - Mode LLM : utilise l'API OpenAI (si cle disponible)
    - Mode regles : genere des rapports bases sur les donnees (pas besoin d'API)

    Pourquoi deux modes ?
    → L'etudiant n'a pas forcement de cle API OpenAI.
       Le mode regles permet de demontrer les fonctionnalites
       meme sans cle API. Le mode LLM donne de meilleurs resultats.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", None)
        self.llm = None
        self.mode = "regles"

        if self.api_key:
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    groq_api_key=self.api_key,
                    temperature=0.7,
                     max_tokens=1000
               )
                self.mode = "llm"
                logger.info("Mode LLM active (Groq - Llama-3.1-8B-Instant)")
            except Exception as e:
                logger.warning(f"Impossible d'initialiser Groq : {e}")
                logger.info("Fallback vers le mode regles")
        else:
            logger.info("Aucune cle API Groq. Mode regles active.")

        # Initialiser MCP
        self.mcp_available = False
        self._init_mcp()


    def _init_mcp(self):
        """Initialise le client MCP pour l'acces responsable aux donnees."""
        try:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from mcp.mcp_simulation import MCPClient, MCPServer, MCPHost

            self.mcp_host = MCPHost()
            self.mcp_client = MCPClient("LLMEnricher")
            self.mcp_server = MCPServer("ScrapingData")

            # Enregistrer les composants
            self.mcp_host.register_client(self.mcp_client)
            self.mcp_host.register_server(self.mcp_server)

            # Permissions
            self.mcp_host.add_permission("LLMEnricher", "read_products")
            self.mcp_host.add_permission("LLMEnricher", "read_stats")

            logger.info("MCP initialise : acces responsable active")
            self.mcp_available = True
        except Exception as e:
            logger.warning(f"MCP non disponible : {e}")
            self.mcp_available = False

    def _call_llm(self, prompt):
        """Appelle le LLM avec un prompt."""
        if self.llm is None:
            return None
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Erreur LLM : {e}")
            return None

    # =========================================================
    # 1. RESUME AUTOMATIQUE DE PRODUITS
    # =========================================================
    def summarize_products(self, df, n_products=10):
        """
        Genere un resume intelligent des meilleurs produits.

        Avec LLM : le LLM lit les descriptions et genere un resume.
        Sans LLM : on genere un resume structure a partir des donnees.
        """
        logger.info(f"Generation de resumes pour les top {n_products} produits...")

        top_products = df.nlargest(n_products, 'final_score')
        summaries = []

        for _, product in top_products.iterrows():
            if self.mode == "llm":
                summary = self._summarize_with_llm(product)
            else:
                summary = self._summarize_with_rules(product)
            summaries.append(summary)

        result = pd.DataFrame(summaries)
        logger.info(f"  {len(result)} resumes generes")
        return result

    def _summarize_with_llm(self, product):
        """Resume d'un produit via LLM."""
        description = str(product.get('description_clean', ''))[:500]
        prompt = f"""Tu es un analyste e-commerce. Resume ce produit en 3 phrases :
        - Nom : {product.get('title', 'N/A')}
        - Prix : {product.get('price', 'N/A')}$
        - Categorie : {product.get('product_type', 'N/A')}
        - Boutique : {product.get('store_name', 'N/A')}
        - Description : {description}
        - Score : {product.get('final_score', 'N/A')}

        Resume en francais : qualite du produit, positionnement prix, potentiel commercial."""

        response = self._call_llm(prompt)
        return {
            'product': product.get('title', ''),
            'price': product.get('price', 0),
            'store': product.get('store_name', ''),
            'score': product.get('final_score', 0),
            'summary': response if response else "Resume non disponible"
        }

    def _summarize_with_rules(self, product):
        """Resume d'un produit sans LLM (base sur les donnees)."""
        price = product.get('price', 0)
        discount = product.get('discount_pct', 0)
        available = product.get('available', False)
        score = product.get('final_score', 0)
        store = product.get('store_name', '')
        title = product.get('title', '')

        # Positionnement prix
        if price < 20:
            price_pos = "entree de gamme"
        elif price < 50:
            price_pos = "milieu de gamme"
        elif price < 100:
            price_pos = "haut de gamme accessible"
        else:
            price_pos = "premium"

        # Potentiel
        if score > 0.5:
            potential = "tres fort potentiel commercial"
        elif score > 0.3:
            potential = "bon potentiel commercial"
        else:
            potential = "potentiel modere"

        # Disponibilite
        avail_text = "disponible en stock" if available else "actuellement en rupture"

        # Remise
        promo_text = f"avec une remise de {discount}%" if discount > 0 else "sans promotion active"

        summary = f"{title} ({store}) est un produit {price_pos} a {price:.2f}$ {promo_text}. "
        summary += f"Il est {avail_text} et presente un {potential} (score: {score:.3f})."

        return {
            'product': title,
            'price': price,
            'store': store,
            'score': score,
            'summary': summary
        }

    # =========================================================
    # 2. ANALYSE CONCURRENTELLLE
    # =========================================================
    def competitive_analysis(self, df):
        """
        Genere un rapport d'analyse concurrentielle entre les boutiques.

        Compare les boutiques sur : prix, remise, disponibilite, succes.
        """
        logger.info("Generation de l'analyse concurrentielle...")

        # Statistiques par boutique
        stats = df.groupby('store_name').agg(
            produits=('product_id', 'count'),
            prix_moyen=('price', 'mean'),
            prix_min=('price', 'min'),
            prix_max=('price', 'max'),
            remise_moy=('discount_pct', 'mean'),
            taux_dispo=('is_available', 'mean'),
            taux_succes=('produit_succes', 'mean'),
            score_moyen=('final_score', 'mean'),
            variantes_moy=('variants_count', 'mean')
        ).round(3)

        if self.mode == "llm":
            report = self._competitive_with_llm(stats)
        else:
            report = self._competitive_with_rules(stats, df)

        return report

    def _competitive_with_llm(self, stats):
        """Rapport concurrentiel via LLM."""
        stats_text = stats.to_string()
        prompt = f"""Tu es un consultant e-commerce. Analyse ces statistiques de 4 boutiques en ligne :

{stats_text}

Genere un rapport en francais avec :
1. Un resume executif (3 lignes)
2. Les points forts de chaque boutique
3. Les faiblesses de chaque boutique
4. Les opportunites de marche
5. 3 recommandations strategiques"""

        response = self._call_llm(prompt)
        return response if response else "Rapport non disponible"

    def _competitive_with_rules(self, stats, df):
        """Rapport concurrentiel sans LLM."""
        report = []
        report.append("=" * 60)
        report.append("RAPPORT D'ANALYSE CONCURRENTIELLE")
        report.append(f"Genere le {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)

        # Resume executif
        report.append("\n1. RESUME EXECUTIF")
        report.append("-" * 40)
        best_store = stats['taux_succes'].idxmax()
        cheapest = stats['prix_moyen'].idxmin()
        most_products = stats['produits'].idxmax()
        report.append(f"La boutique la plus performante est {best_store} "
                      f"(taux de succes: {stats.loc[best_store, 'taux_succes']*100:.1f}%).")
        report.append(f"La boutique la plus abordable est {cheapest} "
                      f"(prix moyen: {stats.loc[cheapest, 'prix_moyen']:.2f}$).")
        report.append(f"Le catalogue le plus large appartient a {most_products} "
                      f"({stats.loc[most_products, 'produits']} produits).")

        # Analyse par boutique
        report.append("\n2. ANALYSE PAR BOUTIQUE")
        report.append("-" * 40)
        for store in stats.index:
            s = stats.loc[store]
            report.append(f"\n  {store.upper()}")
            report.append(f"    Produits : {s['produits']}")
            report.append(f"    Prix moyen : {s['prix_moyen']:.2f}$ ({s['prix_min']:.2f}$ - {s['prix_max']:.2f}$)")
            report.append(f"    Remise moyenne : {s['remise_moy']:.1f}%")
            report.append(f"    Disponibilite : {s['taux_dispo']*100:.1f}%")
            report.append(f"    Taux de succes : {s['taux_succes']*100:.1f}%")
            report.append(f"    Score moyen : {s['score_moyen']:.3f}")

            # Points forts
            strengths = []
            if s['taux_succes'] > 0.2:
                strengths.append("bon taux de succes")
            if s['remise_moy'] > 20:
                strengths.append("politique de remise agressive")
            if s['taux_dispo'] > 0.9:
                strengths.append("excellente disponibilite")
            if s['variantes_moy'] > 5:
                strengths.append("large choix de variantes")
            report.append(f"    Points forts : {', '.join(strengths) if strengths else 'a analyser'}")

            # Faiblesses
            weaknesses = []
            if s['taux_succes'] < 0.1:
                weaknesses.append("faible taux de succes")
            if s['taux_dispo'] < 0.7:
                weaknesses.append("stock insuffisant")
            if s['remise_moy'] < 5:
                weaknesses.append("peu de promotions")
            report.append(f"    Faiblesses : {', '.join(weaknesses) if weaknesses else 'aucune critique majeure'}")

        # Opportunites
        report.append("\n3. OPPORTUNITES DE MARCHE")
        report.append("-" * 40)
        # Categories sous-representees
        cat_counts = df['product_type'].value_counts().head(5)
        report.append("  Categories dominantes :")
        for cat, count in cat_counts.items():
            report.append(f"    - {cat} : {count} produits")

        # Prix moyens par categorie
        report.append("\n  Prix moyens par categorie :")
        price_by_cat = df.groupby('product_type')['price'].mean().sort_values(ascending=False).head(5)
        for cat, price in price_by_cat.items():
            report.append(f"    - {cat} : {price:.2f}$")

        # Recommandations
        report.append("\n4. RECOMMANDATIONS STRATEGIQUES")
        report.append("-" * 40)
        report.append("  1. STOCK : Maintenir un taux de disponibilite superieur a 90%")
        report.append("     pour maximiser le taux de succes des produits.")
        report.append("  2. PROMOTIONS : Les remises entre 20% et 40% semblent optimales")
        report.append("     pour attirer les clients sans devaloriser la marque.")
        report.append("  3. DESCRIPTIONS : Enrichir les descriptions produits (mots cles,")
        report.append("     details techniques) pour ameliorer le score de qualite.")

        report.append("\n" + "=" * 60)
        return "\n".join(report)

    # =========================================================
    # 3. RAPPORT DE TENDANCES
    # =========================================================
    def trend_report(self, df):
        """
        Genere un rapport sur les tendances detectees dans les donnees.
        """
        logger.info("Generation du rapport de tendances...")

        if self.mode == "llm":
            return self._trend_with_llm(df)
        else:
            return self._trend_with_rules(df)

    def _trend_with_llm(self, df):
        """Rapport de tendances via LLM."""
        # Statistiques resumees
        stats = {
            'total_produits': len(df),
            'prix_moyen': df['price'].mean(),
            'categories_top': df['product_type'].value_counts().head(5).to_dict(),
            'remise_moyenne': df['discount_pct'].mean(),
            'taux_succes': df['produit_succes'].mean()
        }

        prompt = f"""Tu es un analyste e-commerce. Genere un rapport de tendances base sur ces donnees :

{stats}

Le rapport doit contenir :
1. Les 5 produits/categories tendance
2. Les tendances de prix
3. Les tendances de promotion
4. Les predictions pour les prochaines semaines
5. 3 actions recommandees

Reponds en francais."""

        response = self._call_llm(prompt)
        return response if response else "Rapport non disponible"

    def _trend_with_rules(self, df):
        """Rapport de tendances sans LLM."""
        report = []
        report.append("=" * 60)
        report.append("RAPPORT DE TENDANCES ECOMMERCE")
        report.append(f"Genere le {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)

        # Top categories
        report.append("\n1. CATEGORIES TENDANCE")
        report.append("-" * 40)
        top_cats = df['product_type'].value_counts().head(5)
        for i, (cat, count) in enumerate(top_cats.items(), 1):
            pct = count / len(df) * 100
            report.append(f"  {i}. {cat} : {count} produits ({pct:.1f}%)")

        # Tendances de prix
        report.append("\n2. TENDANCES DE PRIX")
        report.append("-" * 40)
        report.append(f"  Prix moyen global : {df['price'].mean():.2f}$")
        report.append(f"  Prix median : {df['price'].median():.2f}$")
        report.append(f"  Ecart-type : {df['price'].std():.2f}$")
        report.append(f"  Fourchette : {df['price'].min():.2f}$ - {df['price'].max():.2f}$")

        # Par boutique
        for store in df['store_name'].unique():
            store_data = df[df['store_name'] == store]
            report.append(f"  {store} : {store_data['price'].mean():.2f}$ en moyenne")

        # Tendances promotionnelles
        report.append("\n3. TENDANCES PROMOTIONNELLES")
        report.append("-" * 40)
        promo_pct = (df['discount_pct'] > 0).mean() * 100
        report.append(f"  Produits en promotion : {promo_pct:.1f}%")
        report.append(f"  Remise moyenne (produits en promo) : "
                      f"{df[df['discount_pct'] > 0]['discount_pct'].mean():.1f}%")

        # Par boutique
        for store in df['store_name'].unique():
            store_data = df[df['store_name'] == store]
            store_promo = (store_data['discount_pct'] > 0).mean() * 100
            report.append(f"  {store} : {store_promo:.1f}% en promotion")

        # Produits emblematiques
        report.append("\n4. PRODUITS EMBLEMATIQUES")
        report.append("-" * 40)
        top5 = df.nlargest(5, 'final_score')
        for i, (_, p) in enumerate(top5.iterrows(), 1):
            report.append(f"  {i}. {p['title'][:50]} - {p['price']:.2f}$ ({p['store_name']}) "
                          f"[Score: {p['final_score']:.3f}]")

        # Actions recommandees
        report.append("\n5. ACTIONS RECOMMANDEES")
        report.append("-" * 40)
        report.append("  1. SURVEILLER : Les produits a forte remise (>40%) car ils")
        report.append("     peuvent indiquer une fin de serie ou un stock a ecouler.")
        report.append("  2. DEVELOPPER : Les categories sous-representees mais a")
        report.append("     fort taux de succes pour diversifier l'offre.")
        report.append("  3. OPTIMISER : Les descriptions des produits a faible score")
        report.append("     pour ameliorer leur attractivite.")

        report.append("\n" + "=" * 60)
        return "\n".join(report)

    # =========================================================
    # 4. STRATEGIES MARKETING
    # =========================================================
    def marketing_strategies(self, df):
        """
        Propose des strategies marketing basees sur l'analyse des donnees.
        """
        logger.info("Generation des strategies marketing...")

        if self.mode == "llm":
            return self._marketing_with_llm(df)
        else:
            return self._marketing_with_rules(df)

    def _marketing_with_llm(self, df):
        """Strategies marketing via LLM."""
        summary = {
            'prix_moyen': df['price'].mean(),
            'remise_moyenne': df['discount_pct'].mean(),
            'taux_succes': df['produit_succes'].mean(),
            'top_categories': df['product_type'].value_counts().head(5).to_dict(),
            'boutiques': df['store_name'].unique().tolist()
        }

        prompt = f"""Tu es un consultant marketing e-commerce.
Base sur ces donnees de 4 boutiques en ligne :
{summary}

Propose 5 strategies marketing detailrees en francais :
1. Strategie de prix
2. Strategie de promotion
3. Strategie de contenu
4. Strategie de fidélisation
5. Strategie d'expansion

Pour chaque strategie : description, mise en oeuvre, impact attendu."""

        response = self._call_llm(prompt)
        return response if response else "Strategies non disponibles"

    def _marketing_with_rules(self, df):
        """Strategies marketing sans LLM."""
        report = []
        report.append("=" * 60)
        report.append("STRATEGIES MARKETING RECOMMANDEES")
        report.append(f"Genere le {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)

        avg_price = df['price'].mean()
        avg_discount = df['discount_pct'].mean()
        success_rate = df['produit_succes'].mean()

        # Strategie 1 : Prix
        report.append("\n1. STRATEGIE DE PRIX")
        report.append("-" * 40)
        report.append(f"  Prix moyen actuel : {avg_price:.2f}$")
        report.append(f"  Recommandation : Adopter une politique de prix dynamique")
        report.append(f"  - Produits premium (>100$) : maintenir la qualite percue")
        report.append(f"  - Produits milieu (30-100$) : zone optimale de conversion")
        report.append(f"  - Produits entree (<30$) : utiliser comme produits d'appel")
        report.append(f"  Impact attendu : +15% de taux de conversion")

        # Strategie 2 : Promotions
        report.append("\n2. STRATEGIE DE PROMOTION")
        report.append("-" * 40)
        promo_products = (df['discount_pct'] > 0).mean() * 100
        report.append(f"  Produits actuellement en promo : {promo_products:.1f}%")
        report.append(f"  Remise moyenne : {avg_discount:.1f}%")
        report.append(f"  Recommandation :")
        report.append(f"  - Flash sales sur les produits a stock eleve")
        report.append(f"  - Bundles (packs) pour augmenter le panier moyen")
        report.append(f"  - Remises progressives : 10% -> 20% -> 30% selon l'anciennete")
        report.append(f"  Impact attendu : +25% de volume de ventes")

        # Strategie 3 : Contenu
        report.append("\n3. STRATEGIE DE CONTENU")
        report.append("-" * 40)
        avg_desc_len = df['word_count_description'].mean()
        report.append(f"  Longueur moyenne des descriptions : {avg_desc_len:.0f} mots")
        report.append(f"  Recommandation :")
        report.append(f"  - Enrichir les descriptions courtes (<50 mots)")
        report.append(f"  - Ajouter des mots cles SEO")
        report.append(f"  - Inclure des avis clients dans les fiches")
        report.append(f"  - Creer du contenu video pour les produits premium")
        report.append(f"  Impact attendu : +20% de trafic organique")

        # Strategie 4 : Fidelisation
        report.append("\n4. STRATEGIE DE FIDELISATION")
        report.append("-" * 40)
        report.append(f"  Recommandation :")
        report.append(f"  - Programme de points sur les achats repetes")
        report.append(f"  - Acces anticipe aux nouvelles collections")
        report.append(f"  - Remises exclusives pour les clients reguliers")
        report.append(f"  - Notifications personnalisees basees sur l'historique")
        report.append(f"  Impact attendu : +30% de taux de retention")

        # Strategie 5 : Expansion
        report.append("\n5. STRATEGIE D'EXPANSION")
        report.append("-" * 40)
        report.append(f"  Recommandation :")
        report.append(f"  - Diversifier vers les categories sous-representees")
        report.append(f"  - Explorer de nouveaux marches geographiques")
        report.append(f"  - Partenariats avec des influenceurs niche")
        report.append(f"  - Marketplace complementaires (Amazon, eBay)")
        report.append(f"  Impact attendu : +40% de portee marchande")

        report.append("\n" + "=" * 60)
        return "\n".join(report)

    # =========================================================
    # 5. CHATBOT (pour le dashboard)
    # =========================================================
    def chatbot_response(self, question, df):
        """
        Repond a une question sur les donnees.

        Utilise le LLM si disponible, sinon repond avec des regles.
        """
        if self.mode == "llm":
            return self._chatbot_with_llm(question, df)
        else:
            return self._chatbot_with_rules(question, df)

    def generate_scraping_prompt(self, platform):
        """
        Genere un prompt de scraping specifique selon la plateforme.

        Fonctionnalite demandee par le prof : "Generer automatiquement
        des prompts de scraping specifiques selon la plateforme detectee."
        """
        if self.mode == "llm":
            prompt = f"""Tu es un expert en web scraping e-commerce.
Genere un prompt detaille pour scraper des produits sur la plateforme {platform}.
Le prompt doit inclure :
1. L'URL de base a scraper
2. Les selecteurs CSS ou endpoints API a utiliser
3. Les champs a extraire (titre, prix, description, note, avis, images)
4. Les precautions a prendre (rate limiting, User-Agent)
5. Un exemple de code Python

Reponds en francais."""

            response = self._call_llm(prompt)
            return response if response else self._scraping_prompt_rules(platform)
        else:
            return self._scraping_prompt_rules(platform)

    def _scraping_prompt_rules(self, platform):
        """Genere un prompt de scraping sans LLM."""
        prompts = {
            "shopify": """Prompt de scraping pour Shopify :
- URL : https://BOUTIQUE/products.json?page=N&limit=250
- Methode : API JSON via requests
- Champs : id, title, vendor, product_type, variants[0].price, variants[0].available
- Pagination : incrementer page jusqu'a reponse vide
- Delai : 2 secondes entre chaque requete""",
            "woocommerce": """Prompt de scraping pour WooCommerce :
- URL API : https://BOUTIQUE/wp-json/wc/store/products
- Fallback HTML : https://BOUTIQUE/shop/page/N/
- Selecteurs CSS : li.product, .woocommerce-loop-product__title, .price .amount
- Champs : name, price, regular_price, description, images
- Delai : 2 secondes entre chaque requete"""
        }
        return prompts.get(platform, f"Plateforme '{platform}' non reconnue. Plateformes supportees : shopify, woocommerce")

    def clean_with_llm(self, text):
        """
        Utilise le LLM pour nettoyer et uniformiser un texte.

        Fonctionnalite demandee par le prof : "Reformuler ou nettoyer
        les donnees extraites (ex : uniformiser les titres de produits)."
        """
        if not text or pd.isna(text):
            return ""

        if self.mode == "llm":
            prompt = f"""Nettoie et uniformise ce titre de produit e-commerce.
Enleve les caracteres speciaux inutiles, corrige les fautes, uniformise le format.
Reponds UNIQUEMENT avec le titre nettoye, rien d'autre.

Titre original : {text}

Titre nettoye :"""

            response = self._call_llm(prompt)
            return response.strip() if response else text
        else:
            # Mode regles : nettoyage basique
            import re
            clean = re.sub(r'\s+', ' ', str(text)).strip()
            clean = re.sub(r'[^\w\s\-\']', '', clean)
            return clean

    def generate_client_profile(self, df):
        """
        Genere un profil client base sur les produits les plus consultes.

        Fonctionnalite demandee par le prof : "Creer un profil client
        base sur les produits les plus consultes."
        """
        logger.info("Generation du profil client...")

        # Analyse des top produits
        top_products = df.nlargest(50, 'final_score')

        # Categories preferees
        top_cats = top_products['product_type'].value_counts().head(5)

        # Fourchette de prix preferee
        avg_price = top_products['price'].mean()

        # Boutiques preferees
        top_stores = top_products['store_name'].value_counts().head(3)

        if self.mode == "llm":
            context = f"""
Top categories : {top_cats.to_dict()}
Prix moyen prefere : {avg_price:.2f}$
Boutiques preferees : {top_stores.to_dict()}
Nombre de produits analyses : {len(top_products)}
"""
            prompt = f"""Tu es un analyste marketing. Base sur les produits les plus populaires,
genere un profil client type en francais.

Donnees : {context}

Le profil doit contenir :
1. Demographie probable (age, genre, revenu)
2. Preferences de prix
3. Categories preferees
4. Comportement d'achat
5. Recommandations marketing pour ce profil"""

            response = self._call_llm(prompt)
            return response if response else self._client_profile_rules(top_cats, avg_price, top_stores)
        else:
            return self._client_profile_rules(top_cats, avg_price, top_stores)

    def _client_profile_rules(self, top_cats, avg_price, top_stores):
        """Profil client sans LLM."""
        report = []
        report.append("PROFIL CLIENT TYPE")
        report.append("=" * 40)
        report.append(f"\n1. CATEGORIES PREFEREES :")
        for cat, count in top_cats.items():
            report.append(f"   - {cat} : {count} produits populaires")
        report.append(f"\n2. BUDGET MOYEN : {avg_price:.2f}$")
        if avg_price < 30:
            report.append("   Profil : acheteur budget-conscious")
        elif avg_price < 100:
            report.append("   Profil : acheteur milieu de gamme")
        else:
            report.append("   Profil : acheteur premium")
        report.append(f"\n3. BOUTIQUES PREFEREES :")
        for store, count in top_stores.items():
            report.append(f"   - {store} : {count} produits dans le top")
        report.append(f"\n4. RECOMMANDATIONS :")
        report.append("   - Cibler les promotions sur les categories preferees")
        report.append("   - Proposer des bundles dans la fourchette de prix")
        report.append("   - Envoyer des alertes de restock pour les produits en rupture")
        return "\n".join(report)



    def _chatbot_with_llm(self, question, df):
        """Reponse chatbot via LLM avec contexte enrichi."""

        # Statistiques globales
        total = len(df)
        n_vars = len(df.columns)
        boutiques = df['store_name'].unique().tolist()

        # Prix
        prix_moyen = df['price'].mean()
        prix_median = df['price'].median()
        prix_min = df['price'].min()
        prix_max = df['price'].max()

        # Produit le plus cher
        plus_cher = df.nlargest(1, 'price').iloc[0]

        # Produit le moins cher
        moins_cher = df.nsmallest(1, 'price').iloc[0]

        # Top 5 produits par score
        top5 = df.nlargest(5, 'final_score')[['title', 'price', 'store_name', 'final_score', 'discount_pct']].to_dict('records')

        # Categories
        top_cats = df['product_type'].value_counts().head(10).to_dict()

        # Promotions
        n_promo = (df['discount_pct'] > 0).sum()
        pct_promo = n_promo / total * 100
        remise_moy = df[df['discount_pct'] > 0]['discount_pct'].mean()
        plus_grosse_remise = df.nlargest(1, 'discount_pct').iloc[0]

        # Stock
        n_stock = df['is_available'].sum()
        n_rupture = total - n_stock

        # Succes
        n_succes = df['produit_succes'].sum()
        taux_succes = n_succes / total * 100

        # Stats par boutique
        stats_boutiques = []
        for store in boutiques:
            s = df[df['store_name'] == store]
            stats_boutiques.append({
                'nom': store,
                'produits': len(s),
                'prix_moyen': round(s['price'].mean(), 2),
                'remise_moy': round(s['discount_pct'].mean(), 1),
                'taux_succes': round(s['produit_succes'].mean() * 100, 1),
                'dispo': round(s['is_available'].mean() * 100, 1)
            })

        # Top produits par boutique
        top_par_boutique = {}
        for store in boutiques:
            s = df[df['store_name'] == store]
            top_par_boutique[store] = s.nlargest(3, 'final_score')[['title', 'price', 'final_score']].to_dict('records')

        # Variables du dataset
        var_list = list(df.columns)

        context = f"""
=== STATISTIQUES GENERALES ===
- Total produits : {total}
- Nombre de variables : {n_vars}
- Variables : {var_list[:20]}... (et {n_vars - 20} autres)
- Boutiques : {boutiques}

=== PRIX ===
- Prix moyen : {prix_moyen:.2f}$
- Prix median : {prix_median:.2f}$
- Prix min : {prix_min:.2f}$ (produit : {moins_cher['title']}, {moins_cher['store_name']})
- Prix max : {prix_max:.2f}$ (produit : {plus_cher['title']}, {plus_cher['store_name']})

=== TOP 5 PRODUITS (par score) ===
{chr(10).join([f"- {p['title']} | {p['price']}$ | {p['store_name']} | Score: {p['final_score']}" for p in top5])}

=== CATEGORIES ===
{chr(10).join([f"- {cat}: {count} produits" for cat, count in top_cats.items()])}

=== PROMOTIONS ===
- Produits en promotion : {n_promo} ({pct_promo:.1f}%)
- Remise moyenne (promos) : {remise_moy:.1f}%
- Plus grosse remise : {plus_grosse_remise['discount_pct']}% ({plus_grosse_remise['title']}, {plus_grosse_remise['store_name']})

=== STOCK ===
- En stock : {n_stock} ({n_stock/total*100:.1f}%)
- En rupture : {n_rupture} ({n_rupture/total*100:.1f}%)

=== SUCCES ===
- Produits succes : {n_succes} ({taux_succes:.1f}%)
- Variable cible : produit_succes (Top 20% par score)

=== PAR BOUTIQUE ===
{chr(10).join([f"- {s['nom']}: {s['produits']} produits, prix moy: {s['prix_moyen']}$, remise: {s['remise_moy']}%, succes: {s['taux_succes']}%, dispo: {s['dispo']}%" for s in stats_boutiques])}

=== TOP PRODUITS PAR BOUTIQUE ===
{chr(10).join([f"{store}:" + chr(10) + chr(10).join([f"  - {p['title']} ({p['price']}$, score: {p['final_score']})" for p in prods]) for store, prods in top_par_boutique.items()])}

=== VARIABLES CLES ===
- price : prix du produit en dollars
- discount_pct : pourcentage de remise
- final_score : score composite (0-1) combinant prix, dispo, variantes, images, description, remise
- produit_succes : 1 si top 20%, 0 sinon
- is_available : 1 si en stock, 0 si rupture
- store_name : nom de la boutique
- product_type : categorie du produit
"""

        prompt = f"""Tu es un assistant e-commerce expert. Tu reponds en francais.
Tu as acces aux donnees completes de 4 boutiques en ligne ({total} produits).

Voici les donnees :

{context}

Question : {question}

Pour repondre, suis cette methode de raisonnement :
1. D'abord, identifie quelles donnees du contexte sont pertinentes pour la question
2. Ensuite, analyse ces donnees pour extraire les informations cles
3. Enfin, formule une reponse claire et structuree

Reponds de maniere precise, en citant des chiffres et des noms de produits quand c'est pertinent.
Sois concis mais complet. Si la question ne correspond a aucune donnee disponible, dis-le clairement."""

        response = self._call_llm(prompt)
        return response if response else "Je ne peux pas repondre pour le moment."


    def _chatbot_with_rules(self, question, df):
        """Reponse chatbot sans LLM (base sur les mots-cles)."""
        q = question.lower()

        # Nombre de produits
        if any(word in q for word in ['combien', 'nombre', 'total', 'produits']):
            return f"La base contient {len(df)} produits repartis sur {df['store_name'].nunique()} boutiques."

        # Prix
        if any(word in q for word in ['prix', 'cher', 'cout', 'coute', 'coût', 'coûte', 'couter', 'coûter']):
            return (f"Le prix moyen est de {df['price'].mean():.2f}$. "
                    f"Le moins cher : {df['price'].min():.2f}$. "
                    f"Le plus cher : {df['price'].max():.2f}$.")

        # Meilleur produit
        if any(word in q for word in ['meilleur', 'top', 'best', 'premier']):
            top = df.nlargest(1, 'final_score').iloc[0]
            return (f"Le meilleur produit est : {top['title']} "
                    f"({top['store_name']}, {top['price']:.2f}$, score: {top['final_score']:.3f}).")

        # Boutiques
        if any(word in q for word in ['boutique', 'shop', 'magasin', 'store']):
            stores = df['store_name'].value_counts()
            response = "Les boutiques disponibles :\n"
            for store, count in stores.items():
                response += f"  - {store} : {count} produits\n"
            return response

        # Promotion / remise
        if any(word in q for word in ['promo', 'remise', 'discount', 'solde', 'offre']):
            promo = df[df['discount_pct'] > 0]
            return (f"{len(promo)} produits sont en promotion ({len(promo)/len(df)*100:.1f}%). "
                    f"Remise moyenne : {promo['discount_pct'].mean():.1f}%.")

        # Succes
        if any(word in q for word in ['succes', 'succès', 'reussi', 'populaire']):
            success = df[df['produit_succes'] == 1]
            return (f"{len(success)} produits sont consideres comme des succes "
                    f"({len(success)/len(df)*100:.1f}% du catalogue).")

        # Categorie
        if any(word in q for word in ['categorie', 'type', 'genre']):
            cats = df['product_type'].value_counts().head(5)
            response = "Top categories :\n"
            for cat, count in cats.items():
                response += f"  - {cat} : {count} produits\n"
            return response

        # Disponibilite
        if any(word in q for word in ['stock', 'disponible', 'rupture', 'available']):
            available = df['is_available'].sum()
            return (f"{available} produits sont en stock ({available/len(df)*100:.1f}%). "
                    f"{len(df) - available} sont en rupture.")

        # Default
        return (f"Je peux vous renseigner sur : le nombre de produits, les prix, "
                f"les meilleurs produits, les boutiques, les promotions, les categories, "
                f"la disponibilite. Posez-moi une question !")


# =========================================================
# FONCTION UTILITAIRE
# =========================================================
def generate_full_report(df, api_key=None):
    """
    Genere un rapport complet (tous les modules LLM).

    Utile pour le rapport final du projet.
    """
    enricher = LLMEnricher(api_key=api_key)

    print(f"\nMode d'enrichissement : {enricher.mode}")

    # 1. Resumes produits
    print("\n1. Resumes des produits...")
    summaries = enricher.summarize_products(df, n_products=10)
    summaries.to_csv("outputs/product_summaries.csv", index=False)

    # 2. Analyse concurrentielle
    print("\n2. Analyse concurrentielle...")
    competitive = enricher.competitive_analysis(df)
    with open("outputs/competitive_report.txt", "w", encoding="utf-8") as f:
        f.write(competitive)

    # 3. Rapport de tendances
    print("\n3. Rapport de tendances...")
    trends = enricher.trend_report(df)
    with open("outputs/trend_report.txt", "w", encoding="utf-8") as f:
        f.write(trends)

    # 4. Strategies marketing
    print("\n4. Strategies marketing...")
    strategies = enricher.marketing_strategies(df)
    with open("outputs/marketing_strategies.txt", "w", encoding="utf-8") as f:
        f.write(strategies)

    # 5. Profil client
    print("\n5. Profil client type...")
    profile = enricher.generate_client_profile(df)
    with open("outputs/client_profile.txt", "w", encoding="utf-8") as f:
        f.write(profile)

    # 6. Prompt de scraping genere
    print("\n6. Prompts de scraping generes...")
    for platform in ["shopify", "woocommerce"]:
        prompt = enricher.generate_scraping_prompt(platform)
        print(f"  {platform} : {prompt[:80]}...")

    # 7. Nettoyage LLM (exemple)
    print("\n7. Nettoyage LLM des titres...")
    sample = df['title'].head(3).apply(enricher.clean_with_llm)
    print(f"  Exemples nettoyes : {sample.tolist()}")


    print("\nTous les rapports generes dans outputs/")

    return {
        'summaries': summaries,
        'competitive': competitive,
        'trends': trends,
        'strategies': strategies,
        'profile': profile
    }

