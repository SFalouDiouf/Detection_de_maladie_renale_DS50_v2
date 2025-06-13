# ─────────────────────────────────────────────
# 01_Exploration.py  –  Tableau de bord EDA
# ─────────────────────────────────────────────
import sys, pathlib, io, warnings
ROOT = pathlib.Path(__file__).resolve().parents[2]   # racine du repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from st_aggrid import AgGrid, GridOptionsBuilder
from app.components import sidebar, footer
from src import data

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── CONFIG Streamlit ─────────────────────────────────────────
st.set_page_config(page_title="Exploration des données",
                   page_icon="🔍", layout="wide")

sidebar.render()
st.title("Étape 1 – Exploration des données")

# Palette commune
COLORS = {"ckd": "#1f77b4", "notckd": "#ff7f0e"}

# ╭──────────────────────────────────────────╮
# │ 1. UPLOAD (persistant)                   │
# ╰──────────────────────────────────────────╯
with st.expander("📂 Importer un CSV", expanded=True):
    csv = st.file_uploader("Jeu de données au format CSV", type=["csv"])

if csv is not None:
    csv.seek(0)                               # ← reset curseur
    st.session_state["uploaded_csv"] = csv
    df_file = pd.read_csv(csv)
    st.session_state["uploaded_df"] = df_file # ← stocker DF
elif "uploaded_csv" in st.session_state:
    csv = st.session_state["uploaded_csv"]
    csv.seek(0)                               # ← reset curseur
    df_file = pd.read_csv(csv)
    st.session_state["uploaded_df"] = df_file
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

# Harmoniser la cible et convertir les numériques erronés
if "classification" in df.columns:
    df["classification"] = (
        df["classification"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"ckd": "ckd", "notckd": "notckd"})
    )

for col in ["pcv", "wc", "rc"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

st.session_state["clean_df"] = df            # pour d'autres pages
st.session_state["uploaded_df"] = df         # garantie supplémentaire

rows, cols = df.shape
st.success(f"✅ Données chargées : {rows:,} lignes × {cols} colonnes")

# ╭──────────────────────────────────────────╮
# │ 3. KPI                                   │
# ╰──────────────────────────────────────────╯
dupes = df.duplicated().sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Lignes", f"{rows:,}")
k2.metric("Colonnes", f"{cols}")
k3.metric("% de NA", f"{round(raw_df.isna().mean().mean()*100,2)} %")
k4.metric("Duplicats", f"{dupes}")
st.divider()

# ╭──────────────────────────────────────────╮
# │ 3B. Basic Information                    │
# ╰──────────────────────────────────────────╯
st.subheader("🧾 Informations de base sur les données")

with st.expander("👁️‍🗨️ Aperçu brut (df.head)"):
    st.dataframe(df.head())

with st.expander("ℹ️ Informations (.info)"):
    buf = io.StringIO()
    df.info(buf=buf)
    st.code(buf.getvalue(), language="text")

with st.expander("📊 Statistiques descriptives (.describe)"):
    st.dataframe(df.describe(include="all").T)

with st.expander("🎯 Répartition de la variable cible"):
    if "classification" in df.columns:
        counts = df["classification"].value_counts()
        percents = counts / counts.sum() * 100
        st.write("**Valeurs :**")
        st.write(counts)
        st.write("**Pourcentages :**")
        st.write(percents.round(2))

        fig_target = px.bar(
            x=counts.index, y=counts.values,
            labels={"x": "Classe", "y": "Nombre"},
            title="Distribution de la variable cible",
            color=counts.index, color_discrete_map=COLORS
        )
        st.plotly_chart(fig_target, use_container_width=True)
    else:
        st.warning("⚠️ Colonne 'classification' introuvable.")

with st.expander("🔍 Doublons"):
    st.write(f"Nombre de lignes dupliquées : **{dupes}**")

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# PairPlot (échantillonné)
with st.expander("🔍 Analyse croisée PairPlot"):
    st.markdown("Relations entre variables clés selon le statut CKD.")
    sel = ["age", "bp", "sc", "hemo", "bgr", "classification"]
    if all(c in df.columns for c in sel):
        plot_df = df[sel].copy()
        plot_df["classification"] = plot_df["classification"].str.strip().str.lower()
        fig_pp = sns.pairplot(
            plot_df, hue="classification", palette=COLORS,
            diag_kind="kde", corner=True,
            plot_kws={"alpha": 0.6, "s": 30}
        )
        st.pyplot(fig_pp)
    else:
        st.warning("Colonnes nécessaires pour le pairplot indisponibles.")

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

# 4-B  Valeurs manquantes
with tab_na:
    st.subheader("📉 Valeurs manquantes")
    missing_counts = raw_df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values()

    if missing_counts.empty:
        st.success("✅ Aucune valeur manquante détectée.")
    else:
        fig_missing = px.bar(
            x=missing_counts.values, y=missing_counts.index,
            orientation="h",
            title="Nombre de valeurs manquantes par colonne",
            labels={"x": "NAs", "y": "Colonnes"},
            text=missing_counts.values,
            color=missing_counts.values,
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_missing, use_container_width=True)

        st.markdown("> ℹ️ Corrélation entre indicateurs de NA (patrons de co-manquance).")
        na_matrix = raw_df.isna().astype(int)
        corr_na = na_matrix.corr()
        fig_corr_na = px.imshow(
            corr_na, aspect="auto", zmin=0, zmax=1,
            color_continuous_scale="Purples",
            labels=dict(color="Corr NA")
        )
        st.plotly_chart(fig_corr_na, use_container_width=True)

        st.dataframe(missing_counts.to_frame("NAs").sort_values("NAs", ascending=False))

# 4-C  Histogrammes
with tab_hist:
    st.subheader("📊 Histogrammes + KDE")
    st.write("Distributions des variables numériques (couleur = classe).")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        st.info("Aucune variable numérique à afficher.")
    else:
        per_row = 4
        for i, col in enumerate(num_cols):
            if i % per_row == 0:
                cols = st.columns(per_row)
            with cols[i % per_row]:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(
                    data=df, x=col,
                    hue="classification" if "classification" in df.columns else None,
                    kde=True, palette=COLORS, edgecolor="black",
                    alpha=0.75, ax=ax
                )
                ax.set_title(col, fontsize=9)
                ax.set_xlabel(""); ax.set_ylabel("")
                st.pyplot(fig)

# 4-D  Box-plots + % >P99
with tab_box:
    st.subheader("📦 Box-plots (outliers)")
    if not num_cols:
        st.info("Aucune variable numérique.")
    else:
        cols_per_row = 4
        rows_grid = (len(num_cols) + cols_per_row - 1) // cols_per_row
        fig_box = make_subplots(
            rows=rows_grid, cols=cols_per_row,
            subplot_titles=num_cols,
            horizontal_spacing=0.06, vertical_spacing=0.14
        )
        r = c = 1
        for col in num_cols:
            fig_box.add_trace(
                go.Box(
                    x=df[col], name=col, boxpoints="outliers",
                    marker=dict(color="#2ca02c", opacity=0.6),
                    line=dict(color="#2ca02c"),
                    orientation="h", showlegend=False
                ),
                row=r, col=c
            )
            c += 1
            if c > cols_per_row:
                c = 1; r += 1
        fig_box.update_layout(
            height=280 * rows_grid,
            template="plotly_white",
            title="Distribution des variables numériques"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        out_pct = (
            (df[num_cols] > df[num_cols].quantile(0.99))
            .mean()
            .sort_values(ascending=False) * 100
        )
        st.caption("**% de valeurs au-delà du 99ᵉ centile**")
        st.dataframe(out_pct.round(2).to_frame("% > P99"))

# 4-E  Corrélation
with tab_corr:
    st.subheader("🔗 Matrice de corrélation")
    if len(num_cols) < 2:
        st.info("Pas assez de variables numériques.")
    else:
        corr = df[num_cols].corr().round(2)
        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu", zmin=-1, zmax=1,
            labels=dict(color="Corrélation")
        )
        fig_corr.update_layout(title="Corrélations (Pearson)", height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Valeurs de −1 (inverse) à +1 (directe).")

# 4-F  PCA 2D (imputation moyenne rapide)
with tab_pca:
    st.subheader("🌀 Projection PCA (2 composantes)")
    if len(num_cols) < 2:
        st.info("Au moins deux variables numériques requises.")
    else:
        X_num = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
        X_scaled = StandardScaler().fit_transform(X_num)
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])

        if "classification" in df.columns:
            pca_df["Class"] = df["classification"]
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2",
                color="Class", color_discrete_map=COLORS,
                opacity=0.7,
                title="Projection PCA (2 composantes principales)"
            )
        else:
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2",
                opacity=0.7, title="Projection PCA (2 composantes principales)"
            )

        fig_pca.update_layout(height=600)
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption(
            f"Composante 1 : **{pca.explained_variance_ratio_[0]*100:.2f}%**, "
            f"Composante 2 : **{pca.explained_variance_ratio_[1]*100:.2f}%** de la variance."
        )

footer.render()
