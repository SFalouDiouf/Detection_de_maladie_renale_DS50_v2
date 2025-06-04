# --- bootstrap PYTHONPATH ------------------------------
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent      # dossier racine
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------

import streamlit as st
from app.components import sidebar, footer

st.set_page_config(
    page_title="Détection de la maladie rénale chronique",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

sidebar.render()

st.title("🩺Détection de la maladie rénale chronique (CKD)")

st.markdown(
    """
    Cette application interactive vous accompagne dans un projet de **détection de la maladie rénale chronique (Chronic Kidney Disease)** à l’aide de techniques de **Data Science et d’Intelligence Artificielle**.  
    Elle s’appuie sur le jeu de données de [Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease) et a été développée dans le cadre de la formation **Promo DS50**.

    ### 🔍 Étapes disponibles dans le menu latéral :

    1. **Exploration** – Chargement des données, visualisation, valeurs manquantes, outliers  
    2. **Prétraitement** – Nettoyage des données, encodage, normalisation  
    3. **Modélisation** – Entraînement et comparaison de modèles prédictifs  
    4. **Interprétation** – Évaluation finale et visualisation des performances

    ---
    > *Objectif : Créer un pipeline complet, de l’importation des données à la prédiction fiable de la maladie.*
    """
)


footer.render()
