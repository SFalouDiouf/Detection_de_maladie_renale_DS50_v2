# ─────────────────────────────────────────────
# 01_Exploration.py  –  Tableau de bord EDA
# ─────────────────────────────────────────────
import sys, pathlib, io, warnings
ROOT = pathlib.Path(__file__).resolve().parents[2]          # racine du repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objs as go
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

# Palette commune (réutilisée dans tous les graphiques)
COLORS = {"ckd": "#1f77b4", "notckd": "#ff7f0e"}

# ╭──────────────────────────────────────────╮
# │ 1. UPLOAD persistante                    │
# ╰──────────────────────────────────────────╯
with st.expander("📂 Importer un CSV", expanded=True):
    csv = st.file_uploader("Jeu de données au format CSV", type=["csv"])

if csv is not None:
    csv.seek(0)                                   # ← reset curseur
    st.session_state["uploaded_csv"] = csv
    df_file = pd.read_csv(csv)
    st.session_state["uploaded_df"] = df_file
elif "uploaded_csv" in st.session_state:
    csv = st.session_state["uploaded_csv"]
    csv.seek(0)                                   # ← reset curseur
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
st.session_state["raw_df"]    = raw_df.copy()

# Normaliser la cible + convertir les colonnes numériques “texte”
if "classification" in df.columns:
    df["classification"] = (
        df["classification"].astype(str).str.strip().str.lower()
        .replace({"ckd": "ckd", "notckd": "notckd"})
    )
for col in ["pcv", "wc", "rc"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

st.session_state["clean_df"] = df          # pour les pages suivantes
st.session_state["uploaded_df"] = df       # sauvegarde redondante

rows, cols = df.shape
st.success(f"✅ Données chargées : {rows:,} lignes × {cols} colonnes")

# ╭──────────────────────────────────────────╮
# │ 3. KPI header                            │
# ╰──────────────────────────────────────────╯
dupes = df.duplicated().sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Lignes", f"{rows:,}")
k2.metric("Colonnes", f"{cols}")
k3.metric("% de NA", f"{raw_df.isna().mean().mean()*100:.2f}")
k4.metric("Duplicats", f"{dupes}")
st.divider()

# ╭──────────────────────────────────────────╮
# │ 3B. Infos de base                        │
# ╰──────────────────────────────────────────╯
st.subheader("🧾 Informations de base")

with st.expander("👁️‍🗨️ Aperçu (df.head)"):
    st.dataframe(df.head())

with st.expander("ℹ️ df.info()"):
    buf = io.StringIO()
    df.info(buf=buf)
    st.code(buf.getvalue(), language="text")

with st.expander("📊 Statistiques descriptives (.describe)"):
    st.dataframe(df.describe(include="all").T)

with st.expander("🎯 Répartition de la variable cible"):
    if "classification" in df.columns:
        counts = df["classification"].value_counts()
        percents = counts / counts.sum() * 100
        st.write(counts.to_frame("n"))
        st.write(percents.round(2).to_frame("%"))
        st.plotly_chart(
            px.bar(x=counts.index, y=counts.values,
                   labels={"x": "Classe", "y": "Nombre"},
                   color=counts.index, color_discrete_map=COLORS,
                   title="Distribution de la variable cible"),
            use_container_width=True
        )
    else:
        st.warning("Colonne 'classification' introuvable.")

with st.expander("🔍 Doublons"):
    st.write(f"Nombre de lignes dupliquées : **{dupes}**")

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# ╭──────────────────────────────────────────╮
# │ 4. PairPlot                              │
# ╰──────────────────────────────────────────╯
with st.expander("🔍 PairPlot (seaborn)"):
    sel = ["age", "bp", "sc", "hemo", "bgr", "classification"]
    if all(c in df.columns for c in sel):
        pp_df = df[sel].copy()
        pp_df["classification"] = pp_df["classification"].str.strip().str.lower()
        fig_pp = sns.pairplot(pp_df, hue="classification",
                              palette=COLORS, diag_kind="kde",
                              corner=True, plot_kws={"alpha": .6, "s": 30})
        st.pyplot(fig_pp)
    else:
        st.warning("Variables nécessaires absentes pour le pairplot.")

# ╭──────────────────────────────────────────╮
# │ 5. Tabs de visualisation                 │
# ╰──────────────────────────────────────────╯
tab_grid, tab_na, tab_hist, tab_box, tab_corr, tab_pca = st.tabs(
    ["🗒️ Aperçu", "📉 Manquants", "📊 Histogrammes", "📦 Box-plots",
     "🔗 Corrélation", "🌀 PCA"]
)

# 5-A  Ag-Grid
with tab_grid:
    gb = GridOptionsBuilder.from_dataframe(df.head(500))
    gb.configure_pagination(paginationPageSize=20)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)
    AgGrid(df, gridOptions=gb.build(), height=350, theme="streamlit")

# 5-B  Valeurs manquantes
with tab_na:
    st.subheader("📉 Valeurs manquantes & co-manquance")
    miss = raw_df.isna().sum()
    miss = miss[miss > 0].sort_values()
    if miss.empty:
        st.success("✅ Aucune valeur manquante.")
    else:
        st.plotly_chart(
            px.bar(x=miss.values, y=miss.index, orientation="h",
                   title="Nombre de NA par colonne",
                   labels={"x": "NAs", "y": "Colonnes"},
                   color=miss.values, color_continuous_scale="Reds"),
            use_container_width=True
        )
        # heat-map co-manquance
        na_corr = raw_df.isna().astype(int).corr()
        st.plotly_chart(
            px.imshow(na_corr, zmin=0, zmax=1, aspect="auto",
                      color_continuous_scale="Purples",
                      labels=dict(color="Corr NA"),
                      title="Corrélations des indicateurs de NA"),
            use_container_width=True
        )
        st.dataframe(miss.to_frame("NAs").sort_values("NAs", ascending=False))

# 5-C  Histogrammes conditionnels
with tab_hist:
    st.subheader("📊 Histogrammes + KDE par classe")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        per_row = 4
        for i, col in enumerate(num_cols):
            if i % per_row == 0:
                cols = st.columns(per_row)
            with cols[i % per_row]:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(df, x=col,
                             hue="classification" if "classification" in df.columns else None,
                             kde=True, palette=COLORS,
                             edgecolor="black", alpha=.75, ax=ax)
                ax.set_title(col, fontsize=9)
                ax.set_xlabel(""); ax.set_ylabel("")
                st.pyplot(fig)
    else:
        st.info("Pas de variables numériques.")

# 5-D  Box-plots + % > P99
with tab_box:
    st.subheader("📦 Box-plots + quantification des outliers")
    if num_cols:
        per_row = 4
        rows_grid = (len(num_cols) + per_row - 1) // per_row
        fig_box = make_subplots(rows=rows_grid, cols=per_row,
                                subplot_titles=num_cols,
                                horizontal_spacing=0.06,
                                vertical_spacing=0.14)
        r = c = 1
        for col in num_cols:
            fig_box.add_trace(
                go.Box(x=df[col], boxpoints="outliers", orientation="h",
                       marker=dict(color="#2ca02c", opacity=0.6),
                       line=dict(color="#2ca02c"),
                       showlegend=False), row=r, col=c)
            c += 1
            if c > per_row:
                c = 1; r += 1
        fig_box.update_layout(height=280*rows_grid,
                              template="plotly_white",
                              title="Distribution des variables numériques")
        st.plotly_chart(fig_box, use_container_width=True)

        pct_out = (df[num_cols] > df[num_cols].quantile(0.99)).mean()*100
        st.caption("**% de valeurs > P99**")
        st.dataframe(pct_out.round(2).to_frame("% > P99"))
    else:
        st.info("Pas de variables numériques.")

# 5-E  Corrélation
with tab_corr:
    st.subheader("🔗 Corrélation Pearson")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().round(2)
        st.plotly_chart(
            px.imshow(corr, text_auto=True, aspect="auto",
                      color_continuous_scale="RdBu", zmin=-1, zmax=1,
                      labels=dict(color="Corrélation"),
                      title="Matrice de corrélation (numérique)"),
            use_container_width=True
        )
    else:
        st.info("Pas assez de variables numériques.")

# 5-F  PCA (imputation moyenne rapide)
with tab_pca:
    st.subheader("🌀 PCA (2 composantes)")
    if len(num_cols) < 2:
        st.info("Au moins deux variables numériques requises.")
    else:
        X_num = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
        X_scaled = StandardScaler().fit_transform(X_num)
        comps = PCA(n_components=2).fit_transform(X_scaled)
        pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
        if "classification" in df.columns:
            pca_df["Class"] = df["classification"]
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2",
                                 color="Class", color_discrete_map=COLORS,
                                 opacity=.75,
                                 title="Projection PCA (2 composantes)")
        else:
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2",
                                 opacity=.75, title="Projection PCA")
        st.plotly_chart(fig_pca, use_container_width=True)

footer.render()
