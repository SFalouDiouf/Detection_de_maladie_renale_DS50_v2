# ─────────────────────────────────────────────
# 04_Prediction.py
# ─────────────────────────────────────────────
import sys, pathlib, warnings, datetime
import streamlit as st
import pandas as pd, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components import sidebar, footer
warnings.filterwarnings("ignore")

# ╭── CONFIG ──────────────────────────────────╮
st.set_page_config(page_title="🔮 Prédiction CKD",
                   page_icon="🔮", layout="wide")
sidebar.render()
st.title("🔮 Étape 4 — Prédire de nouveaux patients")

# ╭── CONTRÔLE DU MODÈLE ──────────────────────╮
if "best_model" not in st.session_state or "best_threshold" not in st.session_state:
    st.warning("Entraînez d’abord un modèle dans l’onglet « Modélisation ».")
    footer.render()
    st.stop()

best_model     = st.session_state.best_model        # CalibratedClassifierCV
best_threshold = st.session_state.best_threshold    # seuil F1 optimal

# ── Nettoyage identique au pré-traitement ───────────────────
_BIN_MAP = {"yes": 1, "no": 0,
            "present": 1, "notpresent": 0,
            "abnormal": 1, "normal": 0,
            "good": 1, "poor": 0}
_NUM_FORCE = ["age", "bp", "bgr", "bu", "sc", "sod",
              "pot", "hemo", "pcv", "wc", "rc"]

def preprocess_new(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    if "wc" in df.columns:
        df["wc"] = df["wc"].astype(str).str.replace(",", "", regex=False)

    for col in _NUM_FORCE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        df[col] = (df[col].astype(str)
                           .str.strip()
                           .str.lower()
                           .replace(_BIN_MAP))
    return df

# ╭── UPLOAD CSV UTILISATEUR ──────────────────╮
st.subheader("📁 Charger un fichier CSV de nouveaux patients")
file = st.file_uploader("Sélectionnez votre CSV :", type=["csv"])

if file:
    try:
        raw_df = pd.read_csv(file)
        st.write("Aperçu des données importées :", raw_df.head())

        cleaned_df = preprocess_new(raw_df)

        # ↦ le pipeline interne de best_model gère l’encodage/OHE/scale
        probs = best_model.predict_proba(cleaned_df)[:, 1]
        preds = (probs >= best_threshold).astype(int)

        labels = pd.Series(preds).map({0: "Sain", 1: "Malade"})
        result = raw_df.copy()
        result["Probabilité maladie"] = probs.round(3)
        result["Prédiction"]          = labels

        st.success("✅ Prédictions réalisées")
        st.dataframe(result)

        # bouton de téléchargement
        csv_out = result.to_csv(index=False).encode("utf-8")
        ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button("💾 Télécharger les prédictions",
                           data=csv_out,
                           file_name=f"pred_ckd_{ts}.csv",
                           mime="text/csv")

    except Exception as e:
        st.error(f"Erreur durant la prédiction : {e}")

footer.render()
