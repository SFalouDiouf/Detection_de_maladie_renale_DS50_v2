# --- bootstrap PYTHONPATH ------------------------------
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------

import streamlit as st
from app.components import sidebar, footer
from src import model
from sklearn.model_selection import train_test_split
import plotly.express as px

st.set_page_config(page_title="Modélisation", page_icon="🤖", layout="wide")
sidebar.render()
st.title("Étape 2 – Modélisation et validation croisée")

# ---------- CHECK DATA ---------------------------------
if "clean_df" not in st.session_state:
    st.warning("Veuillez d’abord terminer l’étape d’exploration.")
    footer.render()
    st.stop()

df = st.session_state["clean_df"]

# ---------- PARAMÈTRES --------------------------------
st.sidebar.markdown("### Paramètres d'entraînement")
test_size = st.sidebar.slider("Taille du test", 0.1, 0.4, 0.2, 0.05)
target_col = st.sidebar.selectbox("Variable cible", df.columns[::-1])  # présume la cible en bas
n_folds = st.sidebar.radio("Nombre de folds CV", [5, 10], index=0)

# ---------- ENTRAÎNEMENT -------------------------------
if st.button("🚀 Lancer la comparaison de modèles"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    with st.spinner("Entraînement en cours…"):
        cv_res = model.compare_models_with_cv(X_train, y_train, cv_splits=n_folds)

    st.success("Comparaison terminée")

    # ---------- RÉSULTATS ------------------------------
    st.subheader("Scores moyens (validation croisée)")

    metric_choice = st.selectbox(
        "Afficher la métrique",
        ["test_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1"],
        index=0
    )

    fig = px.bar(
        cv_res.sort_values(metric_choice),
        x="modèle",
        y=metric_choice,
        title=f"Comparaison des modèles – {metric_choice}",
        text_auto=".2f"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cv_res.style.format(precision=3))

    # ---------- MEILLEUR MODÈLE ------------------------
    best_pipe = model.select_best_model(cv_res, metric=metric_choice)
    best_pipe.fit(X_train, y_train)

    st.success(f"Meilleur modèle sélectionné ({metric_choice}) entraîné.")
    st.session_state["best_model"] = best_pipe
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

footer.render()
