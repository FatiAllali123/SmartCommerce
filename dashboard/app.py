"""
=============================================================
DASHBOARD BI - Smart eCommerce Intelligence
=============================================================
Lancer : streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Charger les variables d'environnement
load_dotenv()

# ==============================================================
# CONFIGURATION
# ==============================================================
st.set_page_config(
    page_title="Smart eCommerce Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================
# CSS GLOBAL (sidebar + pages)
# ==============================================================
st.markdown("""
<style>
    /* ====== SIDEBAR ====== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    [data-testid="stSidebar"] .stRadio > label {
        color: white !important;
        font-size: 1.1rem;
        font-weight: 600;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 6px;
        color: #e0dce8 !important;
        font-size: 0.95rem;
        transition: all 0.25s ease;
        cursor: pointer;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
        transform: translateX(4px);
    }

    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: transparent;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b8b4c8 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: white !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.15) !important;
    }

    /* ====== MAIN CONTENT ====== */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
        letter-spacing: -0.5px;
    }

    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* KPI Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    div[data-testid="stMetric"] label {
        color: #a8a4b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
    }

    /* Section headers */
    h2, h3 {
        color: #e8e6f0 !important;
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
    }

    /* General page background */
    .stApp {
        background: #0e0e1a;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 8px 20px;
        color: #a8a4b8;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================
# CHARGEMENT DES DONNEES (CACHED)
# ==============================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/products_clean.csv")

@st.cache_data
def load_comparison():
    path = "outputs/model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_rules():
    path = "outputs/association_rules.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_anomalies():
    path = "outputs/anomalies.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ==============================================================
# CACHE DES CALCULS ML (pour la vitesse)
# ==============================================================
@st.cache_data
def compute_scaled(df):
    """Calcule les donnees standardisees (une seule fois)."""
    features = ['price', 'discount_pct', 'variants_count', 'images_count',
                'is_available', 'tags_count', 'word_count_description']
    features = [c for c in features if c in df.columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].fillna(0))
    return X_scaled, features

@st.cache_data
def compute_pca(_X_scaled):
    """Calcule le PCA (une seule fois)."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(_X_scaled)
    return X_pca

@st.cache_data
def compute_kmeans(_X_scaled, n_clusters):
    """Calcule KMeans (cache par nombre de clusters)."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(_X_scaled)
    return labels

@st.cache_data
def compute_dbscan(_X_scaled, eps, min_samples):
    """Calcule DBSCAN (cache par parametres)."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(_X_scaled)
    return labels


# ==============================================================
# CHARGEMENT INITIAL
# ==============================================================
df = load_data()
X_scaled, cluster_features = compute_scaled(df)
X_pca = compute_pca(X_scaled)



with st.sidebar:
    # ==============================================================
    # LOGO / HEADER
    # ==============================================================
    st.markdown("""
    <div style="text-align: center; padding: 30px 15px 20px 15px;">
        <div style="
            width: 70px; height: 70px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            border-radius: 20px;
            display: flex; align-items: center; justify-content: center;
            margin: 0 auto 15px auto;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
            transform: rotate(-5deg);
        ">
            <span style="font-size: 32px; filter: brightness(0) invert(1);">⚡</span>
        </div>
        <div style="
            font-size: 1.4rem; font-weight: 800;
            background: linear-gradient(135deg, #667eea, #f093fb);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px; line-height: 1.2;
        ">
            SmartCommerce
        </div>
        <div style="
            color: #6c6888; font-size: 0.75rem;
            text-transform: uppercase; letter-spacing: 2.5px;
            margin-top: 6px; font-weight: 500;
        ">
            Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent); margin: 5px 0 20px 0;"></div>
    """, unsafe_allow_html=True)

    # ==============================================================
    # NAVIGATION (pleine largeur)
    # ==============================================================
    pages = [
        {"icon": "🏠", "label": "Vue d'ensemble",    "id": "home"},
        {"icon": "🏆", "label": "Top-K Produits",    "id": "topk"},
        {"icon": "📊", "label": "Clusters KMeans",   "id": "clusters"},
        {"icon": "🔍", "label": "Anomalies DBSCAN",  "id": "anomalies"},
        {"icon": "🔗", "label": "Regles association", "id": "rules"},
        {"icon": "🤖", "label": "Comparaison ML",    "id": "ml"},
        {"icon": "💬", "label": "Chatbot IA",         "id": "chatbot"},
    ]

    # CSS pour les boutons pleine largeur
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] button[kind="secondary"],
    div[data-testid="stSidebar"] button {
        width: 100% !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 4px !important;
        color: #b8b4c8 !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
        justify-content: flex-start !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover,
    div[data-testid="stSidebar"] button:hover {
        background: rgba(102, 126, 234, 0.15) !important;
        border-color: rgba(102, 126, 234, 0.3) !important;
        color: white !important;
        transform: translateX(3px) !important;
    }
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Boutons de navigation
    for p in pages:
        if st.button(
            f"{p['icon']}  {p['label']}",
            key=f"nav_{p['id']}",
            use_container_width=True,
            type="primary" if st.session_state.get('current_page') == p['id'] else "secondary"
        ):
            st.session_state['current_page'] = p['id']
            st.rerun()

    # Page par defaut
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'home'

    # Mapper la page
    page_id = st.session_state['current_page']
    page_map = {
        'home': "🏠 Vue d'ensemble",
        'topk': "🏆 Top-K Produits",
        'clusters': "📊 Clusters (KMeans)",
        'anomalies': "🔍 Anomalies (DBSCAN)",
        'rules': "🔗 Regles d'association",
        'ml': "🤖 Comparaison ML",
        'chatbot': "💬 Chatbot IA",
    }
    page = page_map.get(page_id, "🏠 Vue d'ensemble")

    st.markdown("""
    <div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent); margin: 15px 0;"></div>
    """, unsafe_allow_html=True)

    # ==============================================================
    # STATS RAPIDES
    # ==============================================================
    st.markdown(f"""
    <div style="
        padding: 18px 20px;
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
    ">
        <div style="
            color: #667eea; font-weight: 700; font-size: 0.7rem;
            text-transform: uppercase; letter-spacing: 2px; margin-bottom: 14px;
        ">
            Dataset Overview
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span style="color: #6c6888; font-size: 0.82rem;">Produits</span>
            <span style="color: #e0dce8; font-weight: 700; font-size: 0.9rem;">{len(df):,}</span>
        </div>
        <div style="height: 1px; background: rgba(255,255,255,0.06); margin: 8px 0;"></div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span style="color: #6c6888; font-size: 0.82rem;">Boutiques</span>
            <span style="color: #e0dce8; font-weight: 700; font-size: 0.9rem;">{df['store_name'].nunique()}</span>
        </div>
        <div style="height: 1px; background: rgba(255,255,255,0.06); margin: 8px 0;"></div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span style="color: #6c6888; font-size: 0.82rem;">Variables</span>
            <span style="color: #e0dce8; font-weight: 700; font-size: 0.9rem;">{len(df.columns)}</span>
        </div>
        <div style="height: 1px; background: rgba(255,255,255,0.06); margin: 8px 0;"></div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #6c6888; font-size: 0.82rem;">Taux succes</span>
            <span style="color: #2ecc71; font-weight: 700; font-size: 0.9rem;">{df['produit_succes'].mean()*100:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ==============================================================
    # FOOTER
    # ==============================================================
    st.markdown("""
    <div style="
        position: relative; bottom: 0;
        text-align: center; padding: 15px;
        color: #4a4660; font-size: 0.7rem;
        letter-spacing: 0.5px;
    ">
        Powered by ML & LLM<br>
        <span style="color: #667eea;">Groq</span> + <span style="color: #f093fb;">Llama 3.1</span>
    </div>
    """, unsafe_allow_html=True)



# ==============================================================
# PAGE 1 : VUE D'ENSEMBLE
# ==============================================================
if page_id == "home":
    st.markdown('<div class="main-header">Smart eCommerce Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyse intelligente des produits e-commerce avec ML, DM et LLMs</div>', unsafe_allow_html=True)
    st.markdown("")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Produits", f"{len(df):,}")
    with col2:
        st.metric("🏪 Boutiques", df['store_name'].nunique())
    with col3:
        st.metric("💰 Prix Moyen", f"${df['price'].mean():.2f}")
    with col4:
        st.metric("🎯 Taux de Succes", f"{df['produit_succes'].mean()*100:.1f}%")

    st.markdown("")
    st.markdown("")

    # Graphiques
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Produits par boutique")
        store_counts = df['store_name'].value_counts().reset_index()
        store_counts.columns = ['Boutique', 'Nombre']
        fig = px.bar(store_counts, x='Boutique', y='Nombre',
                     color='Boutique',
                     color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                     text='Nombre')
        fig.update_traces(textposition='outside', textfont_size=14, textfont_color='white')
        fig.update_layout(showlegend=False, height=420,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font_color='#a8a4b8',
                         xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                         yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💲 Distribution des prix")
        fig = px.histogram(df, x='price', nbins=40, color='store_name',
                          color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                          labels={'price': 'Prix ($)', 'store_name': 'Boutique'})
        fig.update_layout(height=420, barmode='overlay',
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font_color='#a8a4b8',
                         xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                         yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                         legend=dict(bgcolor='rgba(0,0,0,0)'))
        fig.update_traces(opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Disponibilite")
        avail_counts = df['available'].value_counts().reset_index()
        avail_counts.columns = ['Disponible', 'Nombre']
        avail_counts['Disponible'] = avail_counts['Disponible'].map({True: 'En stock', False: 'Rupture'})
        fig = px.pie(avail_counts, values='Nombre', names='Disponible',
                     color_discrete_sequence=['#2ecc71', '#e74c3c'],
                     hole=0.5)
        fig.update_traces(textinfo='percent+label', textfont_size=13, textfont_color='white')
        fig.update_layout(height=380,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font_color='#a8a4b8',
                         legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏷️ Categories de prix")
        price_cat = df['price_category'].value_counts().reset_index()
        price_cat.columns = ['Categorie', 'Nombre']
        fig = px.pie(price_cat, values='Nombre', names='Categorie',
                     color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
                     hole=0.5)
        fig.update_traces(textinfo='percent+label', textfont_size=13, textfont_color='white')
        fig.update_layout(height=380,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font_color='#a8a4b8',
                         legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    # Tableau stats
    st.markdown("---")
    st.subheader("📋 Statistiques par boutique")
    stats = df.groupby('store_name').agg(
        Produits=('product_id', 'count'),
        Prix_moyen=('price', 'mean'),
        Prix_median=('price', 'median'),
        Remise_moy=('discount_pct', 'mean'),
        Taux_succes=('produit_succes', 'mean')
    ).round(2)
    stats['Taux_succes'] = (stats['Taux_succes'] * 100).round(1).astype(str) + '%'
    stats.columns = ['Produits', 'Prix moyen ($)', 'Prix median ($)', 'Remise moy. (%)', 'Taux succes']
    st.dataframe(stats, use_container_width=True)


# ==============================================================
# PAGE 2 : TOP-K PRODUITS
# ==============================================================
elif page_id == "topk":
    st.markdown('<div class="main-header">🏆 Top-K Produits</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Les produits les mieux classes selon le score composite</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        k = st.slider("Nombre de produits (K)", 5, 50, 20)
    with col2:
        stores = st.multiselect("Boutiques", df['store_name'].unique().tolist(),
                               default=df['store_name'].unique().tolist())
    with col3:
        price_range = st.slider("Prix ($)",
                                float(df['price'].min()),
                                float(df['price'].max()),
                                (float(df['price'].min()), float(df['price'].max())))

    filtered = df[
        (df['store_name'].isin(stores)) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1])
    ]

    top_k = filtered.nlargest(k, 'final_score')

    display_cols = ['title', 'price', 'store_name', 'product_type', 'discount_pct',
                    'available', 'final_score', 'produit_succes']
    display_cols = [c for c in display_cols if c in top_k.columns]

    st.dataframe(
        top_k[display_cols].style.background_gradient(subset=['final_score'], cmap='YlOrRd'),
        use_container_width=True,
        height=500
    )

    st.subheader("📈 Scores des Top-K produits")
    fig = px.bar(top_k.head(20), x='final_score', y='title', color='store_name',
                 orientation='h',
                 color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'},
                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                     font_color='#a8a4b8',
                     xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                     legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================
# PAGE 3 : CLUSTERS
# ==============================================================
elif page_id == "clusters":
    st.markdown('<div class="main-header">📊 Segmentation (KMeans)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Les produits regroupes selon leurs caracteristiques</div>', unsafe_allow_html=True)

    k = st.slider("Nombre de clusters", 2, 8, 6)
    cluster_labels = compute_kmeans(X_scaled, k)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Visualisation 2D (PCA)")
        df_viz = pd.DataFrame({'pca_1': X_pca[:, 0], 'pca_2': X_pca[:, 1], 'cluster': cluster_labels})
        df_viz['title'] = df['title'].values
        df_viz['price'] = df['price'].values
        df_viz['store_name'] = df['store_name'].values

        fig = px.scatter(df_viz, x='pca_1', y='pca_2', color='cluster',
                        hover_data=['title', 'price', 'store_name'],
                        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#2ecc71', '#e74c3c'],
                        opacity=0.6)
        fig.update_layout(height=500,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font_color='#a8a4b8',
                         xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                         yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                         legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Taille des clusters")
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = pd.DataFrame({'Cluster': unique, 'Taille': counts})
        fig = px.bar(cluster_sizes, x='Cluster', y='Taille', color='Cluster',
                     text='Taille',
                     color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#2ecc71', '#e74c3c'])
        fig.update_traces(textposition='outside', textfont_color='white')
        fig.update_layout(height=500, showlegend=False,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font_color='#a8a4b8',
                         xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                         yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig, use_container_width=True)

    # ==============================================================
    # VARIANCE EXPLIQUEE (PCA)
    # ==============================================================
    st.subheader("📐 Variance expliquee (PCA)")
    from sklearn.decomposition import PCA as PCAFull
    pca_full = PCAFull(n_components=min(len(cluster_features), 10))
    pca_full.fit(X_scaled)
    var_df = pd.DataFrame({
        'Composante': [f'PC{i+1}' for i in range(len(pca_full.explained_variance_ratio_))],
        'Variance (%)': (pca_full.explained_variance_ratio_ * 100).round(1),
        'Cumul (%)': (pca_full.explained_variance_ratio_.cumsum() * 100).round(1)
    })
    fig = px.bar(var_df, x='Composante', y='Variance (%)',
                 text='Variance (%)',
                 color_discrete_sequence=['#667eea'])
    fig.add_scatter(x=var_df['Composante'], y=var_df['Cumul (%)'],
                    mode='lines+markers', name='Cumul (%)',
                    line=dict(color='#f093fb', width=2))
    fig.update_traces(textposition='outside', textfont_color='white')
    fig.update_layout(height=350,
                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                     font_color='#a8a4b8',
                     xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                     yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                     legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)


    # Dendrogramme (Clustering hierarchique)
    st.subheader("🌳 Dendrogramme (Clustering hierarchique)")
    if os.path.exists("outputs/figures/dendrogram.png"):
        st.image("outputs/figures/dendrogram.png",
                 caption="Dendrogramme - Fusion hierarchique des clusters")
    else:
        st.info("Lancez d'abord `python main.py` pour generer le dendrogramme.")

    # Profil
    st.subheader("📋 Profil des clusters")

    df_temp = df.copy()
    df_temp['cluster'] = cluster_labels
    cluster_profile = df_temp.groupby('cluster').agg(
        Taille=('product_id', 'count'),
        Prix_moyen=('price', 'mean'),
        Remise_moy=('discount_pct', 'mean'),
        Taux_succes=('produit_succes', 'mean'),
        Variantes_moy=('variants_count', 'mean'),
        Images_moy=('images_count', 'mean')
    ).round(2)
    cluster_profile['Taux_succes'] = (cluster_profile['Taux_succes'] * 100).round(1)
    cluster_profile.columns = ['Taille', 'Prix moyen ($)', 'Remise moy. (%)',
                                'Taux succes (%)', 'Variantes moy.', 'Images moy.']
    st.dataframe(cluster_profile, use_container_width=True)
    

    # Noms des clusters
    st.subheader("🏷️ Interpretation des clusters")
    cluster_names = {}
    for c in sorted(df_temp['cluster'].unique()):
        data = df_temp[df_temp['cluster'] == c]
        avg_price = data['price'].mean()
        avg_discount = data['discount_pct'].mean()
        avg_success = data['produit_succes'].mean()

        if avg_price > 200 and avg_success > 0.5:
            name = "Premium & Succes"
        elif avg_price > 100:
            name = "Haut de gamme"
        elif avg_discount > 30:
            name = "Discount & Solde"
        elif avg_success > 0.2:
            name = "Populaire"
        elif avg_price < 30:
            name = "Entree de gamme"
        else:
            name = "Standard"

        cluster_names[c] = name
        st.markdown(f"**Cluster {c} = \"{name}\"** — "
                    f"{len(data)} produits, prix moy: {avg_price:.0f}$, "
                    f"remise: {avg_discount:.0f}%, succes: {avg_success*100:.0f}%")


    # Radar
    st.subheader("🎯 Comparaison radar")
    radar_data = df_temp.groupby('cluster')[cluster_features].mean()
    radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#2ecc71', '#e74c3c', '#f39c12', '#1abc9c']
    fig = go.Figure()
    for i, cluster_id in enumerate(sorted(df_temp['cluster'].unique())):
        values = radar_norm.loc[cluster_id].tolist()
        values.append(values[0])
        labels = cluster_features + [cluster_features[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels, name=f'Cluster {cluster_id}',
            line=dict(color=colors[i % len(colors)], width=2),
            fill='toself', opacity=0.15
        ))
    fig.update_layout(height=500,
                     polar=dict(
                         bgcolor='rgba(0,0,0,0)',
                         radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
                         angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                     ),
                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                     font_color='#a8a4b8',
                     legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================
# PAGE 4 : ANOMALIES
# ==============================================================
elif page_id == "anomalies":
    st.markdown('<div class="main-header">🔍 Detection d\'anomalies</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">DBSCAN identifie les produits atypiques</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        eps = st.slider("Epsilon (rayon)", 0.5, 3.0, 1.5, 0.1)
    with col2:
        min_samples = st.slider("Min samples", 5, 30, 10)

    dbscan_labels = compute_dbscan(X_scaled, eps, min_samples)

    anomalies_mask = dbscan_labels == -1
    n_anomalies = anomalies_mask.sum()
    n_normal = (~anomalies_mask).sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Anomalies", f"{n_anomalies:,}")
    with col2:
        st.metric("🟢 Normaux", f"{n_normal:,}")
    with col3:
        st.metric("📊 % Anomalies", f"{n_anomalies/len(df)*100:.1f}%")

    # Visualisation
    df_viz = pd.DataFrame({
        'pca_1': X_pca[:, 0], 'pca_2': X_pca[:, 1],
        'Type': ['Anomalie' if a else 'Normal' for a in anomalies_mask],
        'title': df['title'].values,
        'price': df['price'].values,
        'store_name': df['store_name'].values
    })

    fig = px.scatter(df_viz, x='pca_1', y='pca_2',
                     color='Type',
                     color_discrete_map={'Anomalie': '#e74c3c', 'Normal': '#4facfe'},
                     hover_data=['title', 'price', 'store_name'],
                     opacity=0.7)
    fig.update_layout(height=500,
                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                     font_color='#a8a4b8',
                     xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                     yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                     legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

    # Tableau anomalies
    st.subheader("📋 Liste des anomalies")
    anomalies_df = df[anomalies_mask].copy()
    if len(anomalies_df) > 0:
        st.dataframe(
            anomalies_df[['title', 'price', 'store_name', 'product_type', 'discount_pct']].head(50),
            use_container_width=True
        )
    else:
        st.info("Aucune anomalie detectee avec ces parametres.")


# ==============================================================
# PAGE 5 : REGLES D'ASSOCIATION
# ==============================================================
elif page_id == "rules":
    st.markdown('<div class="main-header">🔗 Regles d\'association</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Decouvrir quels elements vont ensemble</div>', unsafe_allow_html=True)

    rules = load_rules()

    if rules is not None and len(rules) > 0:
        rules['antecedent_str'] = rules['antecedents'].astype(str)
        rules['consequent_str'] = rules['consequents'].astype(str)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📐 Total regles", f"{len(rules):,}")
        with col2:
            if 'source' in rules.columns:
                st.metric("📂 Sources", rules['source'].nunique())
        with col3:
            st.metric("🚀 Lift max", f"{rules['lift'].max():.2f}")

        min_lift = st.slider("Lift minimum", 1.0, float(rules['lift'].max()), 1.5, 0.1)
        filtered_rules = rules[rules['lift'] >= min_lift]

        display_cols = ['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']
        if 'source' in rules.columns:
            display_cols.append('source')

        st.dataframe(
            filtered_rules[display_cols].sort_values('lift', ascending=False).head(30),
            use_container_width=True,
            height=500
        )

        st.subheader("📈 Support vs Confidence")
        fig = px.scatter(filtered_rules, x='support', y='confidence',
                        size='lift',
                        color='source' if 'source' in rules.columns else None,
                        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'],
                        size_max=20, opacity=0.7)
        fig.update_layout(height=400,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font_color='#a8a4b8',
                         xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                         yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                         legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune regle disponible.")


# ==============================================================
# PAGE 6 : COMPARAISON ML
# ==============================================================
elif page_id == "ml":
    st.markdown('<div class="main-header">🤖 Comparaison des modeles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Random Forest vs XGBoost</div>', unsafe_allow_html=True)

    comparison = load_comparison()

    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)

        fig = go.Figure()
        metrics = comparison['Metrique'].tolist()
        fig.add_trace(go.Bar(name='Random Forest', x=metrics, y=comparison['Random Forest'],
                            marker_color='#4facfe', text=comparison['Random Forest'].round(4),
                            textposition='outside', textfont_color='white'))
        fig.add_trace(go.Bar(name='XGBoost', x=metrics, y=comparison['XGBoost'],
                            marker_color='#f093fb', text=comparison['XGBoost'].round(4),
                            textposition='outside', textfont_color='white'))
        fig.update_layout(barmode='group', height=450,
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font_color='#a8a4b8',
                         yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 1.1]),
                         legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    # Matrices de confusion
    st.subheader("🔢 Matrices de confusion")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest**")
        if os.path.exists("outputs/figures/confusion_matrix_random_forest.png"):
            st.image("outputs/figures/confusion_matrix_random_forest.png")
    with col2:
        st.markdown("**XGBoost**")
        if os.path.exists("outputs/figures/confusion_matrix_xgboost.png"):
            st.image("outputs/figures/confusion_matrix_xgboost.png")

    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation business")
    st.markdown("""
    **Principaux enseignements :**

    1. **La disponibilite** est le facteur #1 pour predire le succes
    2. **Les remises** jouent un role majeur dans l'attractivite
    3. **Le prix** a un impact modere
    4. **La qualite de la description** influence les ventes
    5. **Le nombre de variantes** attire plus de clients

    **Recommandations :**
    - Maintenir les produits en stock
    - Proposer des promotions strategiques
    - Enrichir les descriptions produits
    - Offrir plus de variantes (couleurs, tailles)
    """)







# ==============================================================
# PAGE 7 : CHATBOT
# ==============================================================
elif page_id == "chatbot":
    st.markdown('<div class="main-header">💬 Chatbot IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Posez vos questions sur les donnees ecommerce</div>', unsafe_allow_html=True)

    # Initialiser le chatbot
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llm.enrichment import LLMEnricher

    api_key = None
    enricher = LLMEnricher(api_key=api_key)

    st.info(f"Mode : **{enricher.mode}** | "
            f"Pour activer le mode LLM, definissez la variable d'environnement GROQ_API_KEY dans le fichier .env")


    # Questions suggerees
    st.subheader("Questions suggerees")
    col1, col2 = st.columns(2)
    suggested_questions = [
        "Combien de produits avez-vous ?",
        "Quels sont les meilleurs produits ?",
        "Quelles sont les promotions actuelles ?",
        "Quelles boutiques sont disponibles ?",
        "Quelles sont les categories de produits ?",
        "Combien de produits sont en stock ?"
    ]

    selected_question = None
    for i, q in enumerate(suggested_questions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                selected_question = q

    st.markdown("---")

    # Zone de chat
    st.subheader("Votre question")
    user_input = st.text_input("Tapez votre question ici :", key="chatbot_input")

    question = selected_question or user_input

    if question:
        with st.spinner("Recherche en cours..."):
            response = enricher.chatbot_response(question, df)

        st.markdown("---")
        st.subheader("Reponse")
        st.markdown(f"**Question :** {question}")
        st.markdown(f"**Reponse :** {response}")

    

        # ==============================================================
    # RAPPORTS PRE-GENERES
    # ==============================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 15px 0;">
        <span style="font-size: 1.5rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Rapports pre-generes par IA
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Selection du rapport
    report_type = st.selectbox(
        "Choisir un rapport :",
        [
            "📊 Analyse concurrentielle",
            "📈 Rapport de tendances",
            "💡 Strategies marketing",
            "📝 Resumes produits"
        ]
    )

    # Bouton de generation
    if st.button("🔄 Generer le rapport", use_container_width=True, type="primary"):
        with st.spinner("Generation du rapport en cours..."):
            if report_type == "📊 Analyse concurrentielle":
                report = enricher.competitive_analysis(df)
                st.session_state['report_concurrentielle'] = report
            elif report_type == "📈 Rapport de tendances":
                report = enricher.trend_report(df)
                st.session_state['report_tendances'] = report
            elif report_type == "💡 Strategies marketing":
                report = enricher.marketing_strategies(df)
                st.session_state['report_marketing'] = report

            elif report_type == "📝 Resumes produits":
                summaries = enricher.summarize_products(df, n_products=10)
                st.session_state['report_resumes'] = summaries

    # Affichage du rapport selectionne
    st.markdown("")

    if report_type == "📊 Analyse concurrentielle":
        if 'report_concurrentielle' in st.session_state:
            report = st.session_state['report_concurrentielle']
            sections = report.split("\n\n")
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                if section.startswith("="):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                    padding: 20px; border-radius: 12px; margin: 15px 0;
                    border-left: 4px solid #667eea;">
                        <h3 style="color: white; margin: 0;">{section.replace('=', '').strip()}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                elif section.startswith("-") * 20:
                    st.markdown("")
                else:
                    lines = section.split("\n")
                    formatted = ""
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4."):
                            formatted += f"**{line}**\n\n"
                        elif line.upper() == line and len(line) > 3:
                            formatted += f"### {line}\n\n"
                        elif line.startswith("  -") or line.startswith("  *"):
                            formatted += f"- {line.strip().lstrip('-* ')}\n"
                        elif line.startswith("Points forts") or line.startswith("Faiblesses"):
                            formatted += f"**{line}**\n"
                        else:
                            formatted += f"{line}\n\n"
                    st.markdown(formatted)
        else:
            st.info("Cliquez sur 'Generer le rapport' pour afficher l'analyse concurrentielle.")

    elif report_type == "📈 Rapport de tendances":
        if 'report_tendances' in st.session_state:
            report = st.session_state['report_tendances']
            sections = report.split("\n\n")
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                if section.startswith("="):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                    padding: 20px; border-radius: 12px; margin: 15px 0;
                    border-left: 4px solid #f093fb;">
                        <h3 style="color: white; margin: 0;">{section.replace('=', '').strip()}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    lines = section.split("\n")
                    formatted = ""
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if line[0:2] in ["1.", "2.", "3.", "4.", "5."]:
                            formatted += f"**{line}**\n\n"
                        elif line.startswith("  -") or line.startswith("  *"):
                            formatted += f"- {line.strip().lstrip('-* ')}\n"
                        else:
                            formatted += f"{line}\n\n"
                    st.markdown(formatted)
        else:
            st.info("Cliquez sur 'Generer le rapport' pour afficher le rapport de tendances.")

    elif report_type == "💡 Strategies marketing":
        if 'report_marketing' in st.session_state:
            report = st.session_state['report_marketing']
            sections = report.split("\n\n")
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                if section.startswith("="):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                    padding: 20px; border-radius: 12px; margin: 15px 0;
                    border-left: 4px solid #2ecc71;">
                        <h3 style="color: white; margin: 0;">{section.replace('=', '').strip()}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                elif "STRATEGIE" in section.upper() or "Stratégie" in section:
                    lines = section.split("\n")
                    title = lines[0] if lines else ""
                    st.markdown(f"""
                    <div style="background: rgba(46, 204, 113, 0.1);
                    padding: 15px 20px; border-radius: 10px; margin: 10px 0;
                    border: 1px solid rgba(46, 204, 113, 0.3);">
                        <h4 style="color: #2ecc71; margin: 0 0 10px 0;">{title}</h4>
                    """, unsafe_allow_html=True)
                    formatted = ""
                    for line in lines[1:]:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("*") and "**" in line:
                            clean = line.replace("*", "").strip()
                            formatted += f"**{clean}**\n\n"
                        elif line.startswith("+") or line.startswith("  +"):
                            formatted += f"- {line.strip().lstrip('+ ')}\n"
                        else:
                            formatted += f"{line}\n\n"
                    st.markdown(formatted)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    lines = section.split("\n")
                    formatted = ""
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("+") or line.startswith("  +"):
                            formatted += f"- {line.strip().lstrip('+ ')}\n"
                        else:
                            formatted += f"{line}\n\n"
                    st.markdown(formatted)
        else:
            st.info("Cliquez sur 'Generer le rapport' pour afficher les strategies marketing.")

    elif report_type == "📝 Resumes produits":
        if 'report_resumes' in st.session_state:
            summaries = st.session_state['report_resumes']
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
            padding: 20px; border-radius: 12px; margin: 15px 0;
            border-left: 4px solid #4facfe;">
                <h3 style="color: white; margin: 0;">Resumes des {len(summaries)} meilleurs produits</h3>
            </div>
            """, unsafe_allow_html=True)
            for i, (_, row) in enumerate(summaries.iterrows(), 1):
                with st.expander(f"#{i} - {row.get('product', 'Produit')[:60]} ({row.get('price', 0):.2f}$)", expanded=(i <= 3)):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prix", f"{row.get('price', 0):.2f}$")
                    with col2:
                        st.metric("Boutique", row.get('store', 'N/A'))
                    with col3:
                        st.metric("Score", f"{row.get('score', 0):.3f}")
                    st.markdown(f"**Resume :** {row.get('summary', 'Non disponible')}")
        else:
            st.info("Cliquez sur 'Generer le rapport' pour afficher les resumes produits.")
