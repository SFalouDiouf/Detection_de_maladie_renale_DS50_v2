# --- bootstrap PYTHONPATH ------------------------------
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------

import streamlit as st
from app.components import sidebar, footer
from src import model
import pandas as pd
import numpy as np

st.set_page_config(page_title="Prédiction utilisateur", page_icon="🔍", layout="wide")
sidebar.render()

def preprocess_input(df):
    # Nettoyage général des colonnes binaires
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().replace({
            'yes': 1, 'no': 0,
            'present': 1, 'notpresent': 0,
            'abnormal': 1, 'normal': 0,
            'good': 1, 'poor': 0,
            'nan': np.nan
        })
    # Gestion des valeurs manquantes
    df.fillna(method="ffill", inplace=True)  # ou .fillna(0), ou mediane
    return df

st.title("Étape 3 – Prédiction sur de nouvelles données")

if "best_model" not in st.session_state or "X_train" not in st.session_state:
    st.warning("Veuillez d'abord entraîner un modèle dans l'étape précédente.")
    st.stop()

best_model = st.session_state["best_model"]
X_train = st.session_state["X_train"]

# --- Interface utilisateur ---
st.subheader("📁 Importer un fichier CSV")
user_input = st.file_uploader("Choisissez un fichier contenant de nouvelles observations :", type=["csv"])

if user_input:
    try:
        new_data = pd.read_csv(user_input)

        # Prétraitement identique à celui utilisé lors de l'entraînement
        new_data_cleaned = preprocess_input(new_data)

        # Encoding et alignement des colonnes
        new_data_encoded = pd.get_dummies(new_data_cleaned)
        new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

        # Prédiction
        prediction = best_model.predict(new_data_encoded)
        # Revenir à "sain" / "malade"
        if "label_encoder" in st.session_state:
            prediction = st.session_state["label_encoder"].inverse_transform(prediction)

        st.success("✅ Prédiction réalisée avec succès.")
        st.write("Résultat(s) :", prediction)

        # Optionnel : afficher probabilité si dispo
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(new_data_encoded)
            st.write("🔢 Probabilités :", proba)

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

footer.render()
