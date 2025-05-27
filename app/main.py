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

st.title("Détection de la maladie rénale chronique (CKD)")

st.markdown(
    """
    Bienvenue dans l'application !  
    Utilisez le menu latéral pour naviguer entre les étapes :

    1. **Exploration** – Chargement et exploration du jeu de données  
    2. **Modélisation** – Entraînement et validation des modèles  
    3. **Interprétation** – Résultats et interprétation
    """
)

footer.render()
