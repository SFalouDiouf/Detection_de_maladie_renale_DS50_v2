# ─────────────────────────────────────────────
# 01_Exploration.py  –  Tableau de bord EDA
# ─────────────────────────────────────────────
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from app.components import sidebar, footer
from src import data

import pandas as pd, numpy as np, plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from st_aggrid import AgGrid, GridOptionsBuilder

# ─── CONFIG Streamlit (doit être le 1er appel) ─────────────────
st.set_page_config(page_title="Exploration des données",
                   page_icon="🔍", layout="wide")

sidebar.render()
st.title("Étape 1 – Exploration des données")

# ╭──────────────────────────────────────────╮
# │ 1. UPLOAD  (persistant)                  │
# ╰──────────────────────────────────────────╯
with st.expander("📂 Importer un CSV", expanded=True):
    csv = st.file_uploader("Jeu de données au format CSV", type=["csv"])

if csv is not None:
    st.session_state["uploaded_csv"] = csv
    df = pd.read_csv(csv)
    st.session_state["uploaded_df"] = df  # ✅ stocker le DataFrame
elif "uploaded_csv" in st.session_state:
    csv = st.session_state["uploaded_csv"]
    df = pd.read_csv(csv)
    st.session_state["uploaded_df"] = df  # ✅ pour le cas de rechargement
else:
    st.info("Veuillez importer un fichier pour commencer.")
    footer.render()
    st.stop()


# ╭──────────────────────────────────────────╮
# │ 2. LOAD & CLEAN (cache)                  │
# ╰──────────────────────────────────────────╯
@st.cache_data(show_spinner=False)
def load_and_clean(file):
    raw = data.load_dataset(file)
    clean = data.clean_data(raw)
    return raw, clean

raw_df, df = load_and_clean(csv)
st.session_state["clean_df"] = df

rows, cols = df.shape
st.success(f"✅  Données chargées : {rows:,} lignes × {cols} colonnes")

# ╭──────────────────────────────────────────╮
# │ 3. KPI                                   │
# ╰──────────────────────────────────────────╯
k1, k2, k3 = st.columns(3)
k1.metric("Lignes", f"{rows:,}")
k2.metric("Colonnes", f"{cols}")
k3.metric("% de NA", f"{round(raw_df.isna().mean().mean()*100,2)} %")
st.divider()

# ╭──────────────────────────────────────────╮
# │ 3B. Basic Information                    │
# ╰──────────────────────────────────────────╯
st.subheader("🧾 Informations de base sur les données")

with st.expander("👁️‍🗨️ Aperçu brut (df.head)", expanded=False):
    st.dataframe(df.head())

import io

with st.expander("ℹ️ Informations sur le DataFrame (.info)", expanded=False):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.code(info_str, language="text")


with st.expander("📊 Statistiques descriptives (.describe)", expanded=False):
    st.dataframe(df.describe(include='all').transpose())



with st.expander("🎯 Répartition de la variable cible", expanded=False):
    if 'classification' in df.columns:
        df['classification'] = df['classification'].str.strip().str.lower()
        counts = df['classification'].value_counts()
        percents = df['classification'].value_counts(normalize=True) * 100
        st.write("**Valeurs :**")
        st.write(counts)
        st.write("**Pourcentages :**")
        st.write(percents.round(2))

        fig_target = px.bar(
            x=counts.index,
            y=counts.values,
            labels={'x': 'Classe', 'y': 'Nombre'},
            title="Distribution de la variable cible",
            color=counts.index,  # Use class as color key
            color_discrete_map={"ckd": "#1f77b4", "notckd": "#ff7f0e"}  # assign distinct colors
        )
        st.plotly_chart(fig_target, use_container_width=True)
    else:
        st.warning("⚠️ Colonne 'classification' introuvable.")

with st.expander("🔍 Doublons", expanded=False):
    dupes = df.duplicated().sum()
    st.write(f"Nombre de lignes dupliquées : **{dupes}**")

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

#4-G Pairplots 
with st.expander("🔍 Analyse croisée avec PairPlot (Seaborn)", expanded=False):
    st.markdown("Ce graphique montre les relations entre certaines variables en fonction du statut CKD.")

    # Select a manageable subset of features to plot
    selected_features = ['age', 'bp', 'sc', 'hemo', 'bgr', 'classification']

    if all(col in df.columns for col in selected_features):
        plot_df = df[selected_features].copy()
        plot_df["classification"] = plot_df["classification"].str.strip().str.lower()

        # Plot and display
        fig = sns.pairplot(
            plot_df,
            hue="classification",
            palette={"ckd": "#1f77b4", "notckd": "#ff7f0e"},
            diag_kind="kde",
            corner=True,
            plot_kws={"alpha": 0.6, "s": 30}
        )
        st.pyplot(fig)
    else:
        st.warning("Les colonnes nécessaires pour le pairplot ne sont pas toutes disponibles.")
        
# ╭──────────────────────────────────────────╮
# │ 4. TABS VISU                            │
# ╰──────────────────────────────────────────╯
tab_grid, tab_na, tab_hist, tab_box, tab_corr, tab_pca = st.tabs(
    ["🗒️ Aperçu", "📉 Manquants", "📊 Histogrammes", "📦 Box-plots",
     "🔗 Corrélation", "🌀 PCA"]
)

# 4-A  Aperçu Ag-Grid
with tab_grid:
    gb = GridOptionsBuilder.from_dataframe(df.head(500))
    gb.configure_pagination(paginationPageSize=20)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)
    AgGrid(df, gridOptions=gb.build(), height=350, theme="streamlit")

# 4-B  Heatmap NA
with tab_na:
    st.subheader("📉 Valeurs manquantes – visualisation")

    missing_counts = raw_df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)

    if not missing_counts.empty:
        fig_missing = px.bar(
            x=missing_counts.values,
            y=missing_counts.index,
            orientation='h',
            title="Nombre de valeurs manquantes par colonne",
            labels={"x": "Nombre de valeurs manquantes", "y": "Colonnes"},
            text=missing_counts.values,
            color=missing_counts.values,
            color_continuous_scale="Reds"
        )
        fig_missing.update_layout(height=600)
        st.plotly_chart(fig_missing, use_container_width=True)

        # Optional: Add a summary table below
        st.dataframe(missing_counts.to_frame(name="NAs").sort_values("NAs", ascending=False))
    else:
        st.success("✅ Aucune valeur manquante détectée.")


# 4-C  Histogrammes + KDE (grille 4 colonnes)
with tab_hist:
    st.subheader("📊 Histogrammes + KDE")
    st.write("Visualisation des distributions des variables numériques pour détecter asymétrie, outliers et tendances.")

    # ✅ Define numeric columns here
    num_cols = df.select_dtypes(include=["float64", "int64", "int32"]).columns.tolist()

    if not num_cols:
        st.info("Aucune variable numérique à afficher.")
    else:
        cols_per_row = 4
        total = len(num_cols)

        for i in range(0, total, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < total:
                    col = num_cols[i + j]
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.histplot(df[col].dropna(), kde=True, ax=ax,
                                     color="steelblue", edgecolor="black")
                        ax.set_title(f"{col}", fontsize=10)
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        ax.tick_params(axis='both', labelsize=8)
                        st.pyplot(fig)
                        
# 4-D  Box-plots (grille 4 colonnes)
with tab_box:
    st.subheader("📦 Box-plots (détection des valeurs aberrantes)")
    if not num_cols:
        st.info("Aucune variable numérique.")
    else:
        cols_per_row = 4
        rows_grid = (len(num_cols) + cols_per_row - 1) // cols_per_row

        fig_box = make_subplots(
            rows=rows_grid, cols=cols_per_row,
            subplot_titles=[f"{col}" for col in num_cols],
            horizontal_spacing=0.06,
            vertical_spacing=0.14
        )

        r, c = 1, 1
        for col in num_cols:
            fig_box.add_trace(
                go.Box(
                    x=df[col],
                    name=col,
                    boxpoints="outliers",
                    marker=dict(color="#2ca02c", opacity=0.6),
                    line=dict(color="#2ca02c"),
                    orientation="h",
                    showlegend=False,
                    hovertemplate=f"<b>{col}</b><br>Valeur: %{{x}}<extra></extra>"
                ),
                row=r, col=c
            )

            c += 1
            if c > cols_per_row:
                c = 1
                r += 1

        fig_box.update_layout(
            height=280 * rows_grid,
            template="plotly_white",
            title_text="Distribution des variables numériques avec détection des outliers",
            margin=dict(t=50, b=30),
        )
        fig_box.update_xaxes(title_text="Valeurs")
        fig_box.update_yaxes(showticklabels=False)

        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("Les points en dehors des boîtes représentent des valeurs potentiellement aberrantes (outliers).")


# 4-E  Corrélation
with tab_corr:
    st.subheader("🔗 Matrice de corrélation")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().round(2)

        fig_corr = px.imshow(
            corr,
            text_auto=True,  # show values in cells
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            labels=dict(color="Corrélation")
        )
        fig_corr.update_layout(
            title="Corrélations entre les variables numériques",
            height=600,
            margin=dict(t=50, l=10, r=10, b=40)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.caption("Corrélation de Pearson : -1 (négative) à +1 (positive). Les valeurs diagonales sont toujours 1.")
    else:
        st.info("Pas assez de variables numériques pour afficher une matrice de corrélation.")


# 4-F  PCA 2D
with tab_pca:
    st.subheader("🌀 Projection PCA (2 composantes)")
    if len(num_cols) < 2:
        st.info("Au moins deux variables numériques requises.")
    else:
        # Normalize before PCA
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(df[num_cols])

        # Apply PCA
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
        
        # Add target/classification color if available
        if "classification" in df.columns:
            pca_df["Class"] = df["classification"].str.strip().str.lower()
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2",
                color="Class",
                color_discrete_map={"ckd": "#1f77b4", "notckd": "#ff7f0e"},
                opacity=0.7,
                title="Projection PCA (2 composantes principales)",
                labels={"PC1": "Composante Principale 1", "PC2": "Composante Principale 2"}
            )
        else:
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2",
                opacity=0.7,
                title="Projection PCA (2 composantes principales)"
            )

        fig_pca.update_layout(height=600)
        st.plotly_chart(fig_pca, use_container_width=True)

        # Explained variance
        st.caption(
            f"🧮 La Composante 1 explique **{pca.explained_variance_ratio_[0]*100:.2f}%** de la variance, "
            f"et la Composante 2 **{pca.explained_variance_ratio_[1]*100:.2f}%**."
        )




footer.render()
