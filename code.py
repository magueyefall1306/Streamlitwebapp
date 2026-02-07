import streamlit as st
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import numpy as np


# CONFIG
st.set_page_config(page_title="Dashboard KPI – DuckDB", layout="wide")
st.title("Dashboard d’analyse des ventes")


# CONTEXTE DES DATASETS (évite NameError)
CONTEXTE = {}


# INITIALISATIONS OBLIGATOIRES
df = None
df_f = None

is_amazon = False
is_bk = False
is_mcd = False
file_type = "Inconnu"


# UPLOAD CSV
uploaded_file = st.file_uploader("Téléverser un fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Lire le contenu brut
        content = uploaded_file.read()
        
        # Détecter et gérer l'encodage
        try:
            decoded_content = content.decode('utf-8-sig')  # Gère le BOM UTF-8
        except:
            try:
                decoded_content = content.decode('latin-1')
            except:
                decoded_content = content.decode('cp1252')
        
        # Créer un StringIO pour pandas
        df = pd.read_csv(StringIO(decoded_content))
        df.columns = df.columns.map(str).str.lower().str.strip()

        st.success(
            f"Fichier chargé avec succès : {len(df)} lignes, {len(df.columns)} colonnes"
        )

    except Exception as e:
        st.error(f"Erreur lors de la lecture : {str(e)}")
        st.stop()


# DÉTECTION DU TYPE DE FICHIER
if df is not None:
    is_amazon = "discounted_price" in df.columns
    is_bk = "attribute" in df.columns and "value" in df.columns
    is_mcd = "product_name" in df.columns and "price" in df.columns

    file_type = (
        "Amazon" if is_amazon
        else "Burger King" if is_bk
        else "McDonald's" if is_mcd
        else "Inconnu"
    )

    st.write("Type de fichier :", file_type)
else:
    st.stop()


# Afficher le contexte
if file_type in CONTEXTE:
    with st.expander("En savoir plus sur ce dataset"):
        st.markdown(CONTEXTE[file_type])


# NETTOYAGE SPÉCIFIQUE AMAZON
if is_amazon:
    for col in ["discounted_price", "actual_price"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if "discount_percentage" in df.columns:
        df["discount_percentage"] = (
            df["discount_percentage"]
            .astype(str)
            .str.replace("%", "", regex=False)
        )
        df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors='coerce')


# NETTOYAGE GÉNÉRAL
for col in df.columns:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors="ignore")


# DUCKDB
con = duckdb.connect(database=":memory:")
con.register("data", df)


# SIDEBAR – FILTRES
st.sidebar.header("Filtres")

# Déterminer la colonne de valeur
if is_amazon:
    value_col = "discounted_price"
else:
    value_col = "value"

df_filtered = df.dropna(subset=[value_col])

# Filtre par valeur
if len(df_filtered) > 0:
    min_v, max_v = float(df_filtered[value_col].min()), float(df_filtered[value_col].max())
    slider = st.sidebar.slider(
        f"Filtrer par {value_col}",
        min_v, max_v,
        (min_v, max_v)
    )
    df_f = df_filtered[
        (df_filtered[value_col] >= slider[0]) &
        (df_filtered[value_col] <= slider[1])
    ]
else:
    st.error("Aucune donnée valide trouvée")
    st.stop()


# Filtre supplémentaire pour BK
if is_bk and "attribute" in df_f.columns:
    attributes = df_f["attribute"].unique().tolist()
    selected_attr = st.sidebar.multiselect(
        "Filtrer par attribut",
        attributes,
        default=attributes[:5] if len(attributes) > 5 else attributes
    )
    if selected_attr:
        df_f = df_f[df_f["attribute"].isin(selected_attr)]


# Filtre supplémentaire pour McDonald's
if is_mcd and "item" in df_f.columns:
    items = df_f["item"].unique().tolist()
    selected_items = st.sidebar.multiselect(
        "Filtrer par item",
        items,
        default=items[:5] if len(items) > 5 else items
    )
    if selected_items:
        df_f = df_f[df_f["item"].isin(selected_items)]

con.unregister("data")
con.register("data", df_f)


# KPI
st.subheader("Indicateurs clés de performance (KPI)")

c1, c2, c3, c4 = st.columns(4)

kpi1 = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
kpi2 = con.execute(f"SELECT AVG({value_col}) FROM data").fetchone()[0]
kpi3 = con.execute(f"SELECT MIN({value_col}) FROM data").fetchone()[0]
kpi4 = con.execute(f"SELECT MAX({value_col}) FROM data").fetchone()[0]

c1.metric("Nb d'entrées", f"{kpi1:,}")
c2.metric("Moyenne", f"{kpi2:,.2f}")
c3.metric("Minimum", f"{kpi3:,.2f}")
c4.metric("Maximum", f"{kpi4:,.2f}")


# VISUALISATIONS 
st.subheader("Visualisations des KPI")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# KPI 1 – Histogramme 
with row1_col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df_f[value_col], bins=30, color="#2ecc71", edgecolor="black", alpha=0.7)
    ax.set_title("Distribution des valeurs", fontsize=14, fontweight='bold')
    ax.set_xlabel(value_col.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel("Fréquence", fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    st.caption("**Interprétation** : Cet histogramme montre la distribution des valeurs, permettant d'identifier les plages de prix/valeurs les plus fréquentes.")

# KPI 2 – Violin plot (plus parlant que boxplot)
with row1_col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    parts = ax.violinplot([df_f[value_col].dropna()], vert=True, showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    
    ax.set_title("Distribution détaillée (Violin Plot)", fontsize=14, fontweight='bold')
    ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=11)
    ax.set_xticks([1])
    ax.set_xticklabels(['Distribution'])
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    st.caption("**Interprétation** : Le violin plot révèle la densité de distribution. La partie la plus large indique où se concentrent la majorité des valeurs.")

# KPI 3 – Courbe cumulative 
with row2_col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_values = df_f[value_col].sort_values().reset_index(drop=True)
    cumsum = sorted_values.cumsum()
    ax.plot(cumsum, color="#e74c3c", linewidth=2.5)
    ax.fill_between(range(len(cumsum)), cumsum, alpha=0.3, color="#e74c3c")
    ax.set_title("Évolution cumulée", fontsize=14, fontweight='bold')
    ax.set_xlabel("Index (entrées triées)", fontsize=11)
    ax.set_ylabel("Valeur cumulée", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=cumsum.iloc[-1]/2, color='green', linestyle='--', alpha=0.5, label='50% du total')
    ax.legend()
    st.pyplot(fig)
    plt.close()
    
    st.caption("**Interprétation** : Cette courbe montre l'accumulation progressive des valeurs. Une pente raide indique une concentration rapide.")

# KPI 4 – Top N bar chart (plus lisible que camembert)
with row2_col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if is_amazon and "category" in df_f.columns:
        top_categories = df_f["category"].value_counts().head(8)
        colors = sns.color_palette("Set3", len(top_categories))
        ax.barh(range(len(top_categories)), top_categories.values, color=colors, edgecolor='black')
        ax.set_yticks(range(len(top_categories)))
        ax.set_yticklabels(top_categories.index, fontsize=9)
        ax.set_title("Top 8 catégories", fontsize=14, fontweight='bold')
        ax.set_xlabel("Nombre de produits", fontsize=11)
        ax.invert_yaxis()
        for i, v in enumerate(top_categories.values):
            ax.text(v, i, f' {v}', va='center', fontsize=9)

    elif is_bk and "attribute" in df_f.columns:
        top_attrs = df_f["attribute"].value_counts().head(8)
        colors = sns.color_palette("Set2", len(top_attrs))
        ax.barh(range(len(top_attrs)), top_attrs.values, color=colors, edgecolor='black')
        ax.set_yticks(range(len(top_attrs)))
        ax.set_yticklabels(top_attrs.index, fontsize=9)
        ax.set_title("Top 8 attributs", fontsize=14, fontweight='bold')
        ax.set_xlabel("Fréquence", fontsize=11)
        ax.invert_yaxis()
        for i, v in enumerate(top_attrs.values):
            ax.text(v, i, f' {v}', va='center', fontsize=9)

    elif is_mcd and "item" in df_f.columns:
        top_items = df_f["item"].value_counts().head(8)
        colors = sns.color_palette("Pastel1", len(top_items))
        ax.barh(range(len(top_items)), top_items.values, color=colors, edgecolor='black')
        ax.set_yticks(range(len(top_items)))
        ax.set_yticklabels(top_items.index, fontsize=9)
        ax.set_title("Top 8 items", fontsize=14, fontweight='bold')
        ax.set_xlabel("Fréquence", fontsize=11)
        ax.invert_yaxis()
        for i, v in enumerate(top_items.values):
            ax.text(v, i, f' {v}', va='center', fontsize=9)
    else:
        top_values = df_f.nlargest(10, value_col)[value_col].reset_index(drop=True)
        ax.bar(range(len(top_values)), top_values, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax.set_title("Top 10 valeurs les plus élevées", fontsize=14, fontweight='bold')
        ax.set_xlabel("Rang", fontsize=11)
        ax.set_ylabel("Valeur", fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    st.pyplot(fig)
    plt.close()
    st.caption("**Interprétation** : Ce graphique identifie les catégories/items dominants, facilitant la compréhension des segments principaux.")


# GRAPHIQUES SUPPLÉMENTAIRES SPÉCIFIQUES
if is_bk or is_mcd:
    st.subheader("Analyse temporelle")
    
    if "date" in df_f.columns:
        df_f['date'] = pd.to_datetime(df_f['date'], errors='coerce')
        df_f['year'] = df_f['date'].dt.year
        if df_f['year'].notna().sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            yearly_data = df_f.groupby('year')[value_col].sum().sort_index()
            bars = ax.bar(yearly_data.index.astype(str), yearly_data.values, 
                         color='#9b59b6', edgecolor='black', alpha=0.7, width=0.6)
            ax.plot(yearly_data.index.astype(str), yearly_data.values, 
                   color='#e74c3c', marker='o', linewidth=2, markersize=8, label='Tendance')
            ax.set_title("Évolution des revenus par année", fontsize=14, fontweight='bold')
            ax.set_xlabel("Année", fontsize=11)
            ax.set_ylabel("Valeur totale (M$)", fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            for i, (idx, v) in enumerate(yearly_data.items()):
                ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            st.pyplot(fig)
            plt.close()
            st.caption("**Interprétation** : Cette visualisation montre la croissance année après année, avec la tendance globale. Utile pour identifier la récupération post-COVID (2021-2023).")
    
    if is_mcd and "heading" in df_f.columns:
        st.subheader("Analyse par type de revenus")
        fig, ax = plt.subplots(figsize=(10, 5))
        revenue_by_type = df_f.groupby('heading')[value_col].sum().sort_values(ascending=False)
        colors =['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        ax.bar(range(len(revenue_by_type)), revenue_by_type.values, 
              color=colors[:len(revenue_by_type)], edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(revenue_by_type)))
        ax.set_xticklabels(revenue_by_type.index, rotation=45, ha='right', fontsize=10)
        ax.set_title("Répartition des revenus par type", fontsize=14, fontweight='bold')
        ax.set_ylabel("Revenus totaux (M$)", fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(revenue_by_type.values):
            ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("**Interprétation** : Comparaison entre les revenus des franchises vs. établissements gérés par l'entreprise. Révèle le modèle économique dominant.")


# ANALYSE STATISTIQUE
st.subheader("Statistiques descriptives")

col_stats1, col_stats2 = st.columns(2)

with col_stats1:
    stats_df = df_f[value_col].describe().to_frame()
    stats_df.columns = [value_col.title()]
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

with col_stats2:
    st.write("**Informations générales**")
    info_data = {
        "Métrique": ["Médiane", "Écart-type", "Variance", "Coefficient de variation"],
        "Valeur": [
            f"{df_f[value_col].median():.2f}",
            f"{df_f[value_col].std():.2f}",
            f"{df_f[value_col].var():.2f}",
            f"{(df_f[value_col].std() / df_f[value_col].mean() * 100):.2f}%"
        ]
    }
    st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)


# APERÇU DES DONNÉES
st.subheader("Aperçu des données filtrées")
col_display1, col_display2 = st.columns([1, 3])
with col_display1:
    n_rows = st.selectbox("Nombre de lignes à afficher", [10, 25, 50, 100], index=0)
st.dataframe(df_f.head(n_rows), use_container_width=True)


# TÉLÉCHARGEMENT
st.subheader("Télécharger les données filtrées")

@st.cache_data
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(df_f)

st.download_button(
    label="Télécharger en CSV",
    data=csv_data,
    file_name=f"donnees_filtrees_{file_type.replace(' ', '_')}.csv",
    mime="text/csv",
)
