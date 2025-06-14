# ─────────────────────────────────────────────
# 04_Prediction.py  –  sortie épurée
# ─────────────────────────────────────────────
import sys, pathlib, warnings, datetime
import streamlit as st
import pandas as pd, numpy as np

# --- bootstrap PYTHONPATH (accès aux composants) --------------
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------------

from app.components import sidebar, footer
warnings.filterwarnings("ignore")

# ╭── CONFIG ───────────────────────────────────────────────────╮
st.set_page_config(page_title="🔮 Prédiction CKD",
                   page_icon="🔮", layout="wide")
sidebar.render()
st.title("🔮 Étape 4 — Prédire de nouveaux patients")

# ╭── CONTRÔLE : modèle & seuil dispos ? ───────────────────────╮
if "best_model" not in st.session_state or "best_threshold" not in st.session_state:
    st.warning("ℹ️ Entraînez d’abord un modèle dans l’onglet « Modélisation ».")
    footer.render()
    st.stop()

best_model     = st.session_state.best_model
best_threshold = st.session_state.best_threshold

# ── même nettoyage que dans Pré-traitement ────────────────────
_BIN = {"yes": 1, "no": 0, "present": 1, "notpresent": 0,
        "abnormal": 1, "normal": 0, "good": 1, "poor": 0}
_NUM = ["age", "bp", "bgr", "bu", "sc", "sod",
        "pot", "hemo", "pcv", "wc", "rc"]

def preprocess_new(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    if "wc" in df.columns:
        df["wc"] = df["wc"].astype(str).str.replace(",", "", regex=False)

    for col in _NUM:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        df[col] = (df[col].astype(str)
                           .str.strip()
                           .str.lower()
                           .replace(_BIN))
    return df

# ╭── UPLOAD CSV UTILISATEUR ───────────────────────────────────╮
st.subheader("📁 Charger un fichier CSV de nouveaux patients")
file = st.file_uploader("Sélectionnez votre CSV :", type=["csv"])

if file:
    try:
        raw_df = pd.read_csv(file)
        st.write("Aperçu des données importées :", raw_df.head())

        clean_df = preprocess_new(raw_df)

        # --- prédictions ----------------------------------------------------
        probs = best_model.predict_proba(clean_df)[:, 1]
        preds = (probs >= best_threshold).astype(int)
        labels = pd.Series(preds).map({0: "Sain", 1: "Malade"})

        # --- constitution du résultat (id + proba + prédiction) ------------
        if "id" in raw_df.columns:
            id_col = raw_df["id"]
        else:                           # au cas où le CSV n’aurait pas d’ID
            id_col = raw_df.index

        result = pd.DataFrame({
            "id": id_col,
            "Probabilité maladie": probs.round(3),
            "Prédiction": labels
        })

        st.success("✅ Prédictions réalisées")
        st.dataframe(result)

        # --- téléchargement -------------------------------------------------
        csv_out = result.to_csv(index=False).encode("utf-8")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button("💾 Télécharger les prédictions",
                           data=csv_out,
                           file_name=f"pred_ckd_{ts}.csv",
                           mime="text/csv")

    except Exception as e:
        st.error(f"Erreur durant la prédiction : {e}")

footer.render()
