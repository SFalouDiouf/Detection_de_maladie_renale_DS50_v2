# --- bootstrap PYTHONPATH ------------------------------
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------

import streamlit as st
from app.components import sidebar, footer
from src import model, viz
import joblib
from io import BytesIO

st.set_page_config(page_title="Interprétation", page_icon="📊", layout="wide")
sidebar.render()

st.title("Étape 3 – Interprétation et résultats")

if "best_model" not in st.session_state:
    st.warning("Veuillez d'abord entraîner un modèle dans l'étape précédente.")
    st.stop()

best_model = st.session_state["best_model"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

st.subheader("Performances sur le jeu de test")
metrics_df = model.evaluate_model(best_model, X_test, y_test)
st.dataframe(metrics_df)

st.subheader("Matrice de confusion")
viz.plot_confusion_matrix(best_model, X_test, y_test)

if st.button("Télécharger le modèle entraîné"):
    buffer = BytesIO()
    joblib.dump(best_model, buffer)
    st.download_button(
        label="Télécharger le modèle .joblib",
        data=buffer.getvalue(),
        file_name="ckd_pipeline.joblib",
        mime="application/octet-stream",
    )

footer.render()
