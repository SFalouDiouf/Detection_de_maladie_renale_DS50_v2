# ─────────────────────────────────────────────
# 01_Exploration.py  –  Tableau de bord EDA
# ─────────────────────────────────────────────
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
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
    st.subheader("Carte des valeurs manquantes")
    fig_na = px.imshow(
        raw_df.isnull(),
        aspect="auto",
        color_continuous_scale=[[0, "white"], [1, "#ff6961"]],
        labels=dict(color="NA")
    )
    fig_na.update_layout(height=500, coloraxis_showscale=False)
    st.plotly_chart(fig_na, use_container_width=True)

# 4-C  Histogrammes + KDE (grille 4 colonnes)
with tab_hist:
    st.subheader("Histogrammes + KDE")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("Aucune variable numérique.")
    else:
        cols_per_row = 4
        rows_grid = (len(num_cols) + cols_per_row - 1) // cols_per_row
        fig = make_subplots(
            rows=rows_grid, cols=cols_per_row,
            subplot_titles=num_cols,
            horizontal_spacing=.06, vertical_spacing=.14
        )

        r = c = 1
        for col in num_cols:
            vals = df[col].dropna().values

            # histogramme (barres jointives)
            fig.add_trace(
                go.Histogram(
                    x=vals, nbinsx=30,
                    marker_color="#1f77b4",
                    opacity=.85,
                    showlegend=False,
                    hovertemplate="%{x}<br>Count : %{y}<extra></extra>"
                ),
                row=r, col=c
            )

            # KDE lissée
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=kde(xs) * len(vals) * (xs[1]-xs[0]),  # remise à l’échelle
                    mode="lines",
                    line=dict(color="#ff7f0e", width=2),
                    showlegend=False
                ),
                row=r, col=c
            )

            c = c + 1 if c < cols_per_row else 1
            if c == 1:
                r += 1

        fig.update_layout(
            height=300 * rows_grid,
            template="plotly_white",
            margin=dict(t=40, b=20),
            bargap=0                                   # barres collées
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

# 4-D  Box-plots (grille 4 colonnes)
with tab_box:
    st.subheader("Box-plots (outliers)")
    if not num_cols:
        st.info("Aucune variable numérique.")
    else:
        cols_per_row = 4
        rows_grid = (len(num_cols) + cols_per_row - 1) // cols_per_row
        fig_box = make_subplots(
            rows=rows_grid, cols=cols_per_row,
            subplot_titles=num_cols,
            horizontal_spacing=.06, vertical_spacing=.14
        )

        r = c = 1
        for col in num_cols:
            fig_box.add_trace(
                go.Box(
                    x=df[col],
                    boxpoints="outliers",
                    marker_color="#2ca02c",
                    line_color="#2ca02c",
                    orientation="h",
                    showlegend=False,
                    hovertemplate="%{x}<extra></extra>"
                ),
                row=r, col=c
            )
            c = c + 1 if c < cols_per_row else 1
            if c == 1:
                r += 1

        fig_box.update_layout(
            height=250 * rows_grid,
            template="plotly_white",
            margin=dict(t=40, b=20)
        )
        fig_box.update_xaxes(showticklabels=False)
        fig_box.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_box, use_container_width=True)

# 4-E  Corrélation
with tab_corr:
    st.subheader("Matrice de corrélation")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig_corr = px.imshow(corr, color_continuous_scale="Viridis", aspect="auto")
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Pas assez de variables numériques.")

# 4-F  PCA 2D
with tab_pca:
    st.subheader("Projection PCA (2 composantes)")
    if len(num_cols) < 2:
        st.info("Au moins deux variables numériques requises.")
    else:
        comps = PCA(n_components=2).fit_transform(df[num_cols])
        pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
        st.plotly_chart(
            px.scatter(pca_df, x="PC1", y="PC2",
                       opacity=.75, color_discrete_sequence=["#1f77b4"]),
            use_container_width=True
        )

footer.render()
