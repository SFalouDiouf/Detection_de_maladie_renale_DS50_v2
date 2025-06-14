# ─────────────────────────────────────────────
# 02_Pretraitement.py – pipeline robuste + UI explicative
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd, numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ╭────────── CONFIG STREAMLIT ──────────╮
st.set_page_config(page_title="🧼 Pré-traitement", page_icon="🧹", layout="wide")
st.title("🧹 Pipeline de Pré-traitement des Données")

# ╭────────── CHARGEMENT DF ──────────╮
if "uploaded_df" not in st.session_state:
    st.warning("Importez d’abord le CSV dans la page « Exploration ». 🚩")
    st.stop()

df_raw = st.session_state.uploaded_df.copy()
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = df_raw.copy()
if "id" in df_raw.columns:
    df_raw.drop(columns=["id"], inplace=True)

df_raw["classification"] = (
    df_raw["classification"].astype(str).str.strip().str.lower()
)

# ╭────────── TRANSFORMER : strip "," ──────────╮
class StripThousands(BaseEstimator, TransformerMixin):
    """Supprime la virgule des milliers et convertit en float."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        return pd.DataFrame(
            X.astype(str).str.replace(",", "", regex=False).astype(float),
            columns=X.columns,
            index=X.index,
        )

# ╭────────── EXPANDER 1 : DIAGNOSTIC NA ──────────╮
with st.expander("🔍 Étape 1 – Diagnostic des valeurs manquantes", expanded=True):
    miss = (
        df_raw.isna().sum()
        .loc[lambda s: s > 0]
        .to_frame("Total NA")
        .assign(**{"% NA": lambda d: (d["Total NA"] / len(df_raw) * 100).round(1)})
        .sort_values("% NA", ascending=False)
    )
    if miss.empty:
        st.success("✅ Aucune valeur manquante.")
    else:
        st.dataframe(miss)
        st.markdown(
            "*Règle :* ≤ 5 % ⇒ imputation simple • > 5 % ⇒ imputation KNN (num) "
            "ou catégorie « missing » (cat)."
        )

# ╭────────── FONCTION D’IMPUTATION ──────────╮
def impute_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # 1) “?” → NaN
    df.replace("?", np.nan, inplace=True)

    # 2) Conversion des colonnes numériques (virgules milliers)
    if "wc" in df.columns:
        df["wc"] = df["wc"].astype(str).str.replace(",", "").replace("nan", np.nan)

    numeric_force = ["age", "bp", "bgr", "bu", "sc", "sod", "pot",
                     "hemo", "pcv", "wc", "rc"]
    for col in numeric_force:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) Binarisation explicite
    mapping = {"yes": 1, "no": 0,
               "present": 1, "notpresent": 0,
               "abnormal": 1, "normal": 0,
               "good": 1, "poor": 0}
    bin_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm",
                "cad", "appet", "pe", "ane"]
    for col in bin_cols:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                              .str.strip()
                              .str.lower()
                              .map(mapping)
                              .replace("nan", np.nan))

    # 4) Imputation
    na_pct = df.isna().mean() * 100
    low_na  = [c for c in df.columns if na_pct[c] <= 5]
    high_na = [c for c in df.columns if na_pct[c] > 5]

    for c in low_na:
        if df[c].dtype.kind in "fi":
            df[c].fillna(df[c].median(), inplace=True)
        else:
            df[c].fillna(df[c].mode().iloc[0], inplace=True)

    num_high = [c for c in high_na if df[c].dtype.kind in "fi"]
    cat_high = [c for c in high_na if c not in num_high]

    if num_high:
        df[num_high] = KNNImputer(n_neighbors=3).fit_transform(df[num_high])
    for c in cat_high:
        df[c] = df[c].fillna("missing")

    # 5) Cible binaire
    df["classification"] = df["classification"].replace(
        {"ckd": 1, "ckd\t": 1, "notckd": 0}
    )
    return df

# ╭────────── EXPANDER 2 : IMPUTATION ──────────╮
with st.expander("🛠️ Étape 2 – Nettoyage & imputation", expanded=False):
    st.markdown(
        """
        *Actions :*  
        1. Normaliser les valeurs binaires, supprimer les virgules milliers  
        2. Appliquer la stratégie d’imputation définie à l’étape 1  
        """
    )
    if st.button("🔁 Lancer l’imputation"):
        df_imp = impute_df(df_raw)
        st.session_state.df_imputed = df_imp
        st.success("Imputation terminée ✅")
        st.dataframe(df_imp.head())

# ╭────────── EXPANDER 3 : PIPELINE ──────────╮
with st.expander("⚙️ Étape 3 – Construction du pipeline Scaler + OHE", expanded=False):
    if "df_imputed" not in st.session_state:
        st.info("Veuillez d’abord lancer l’imputation.")
    else:
        df_imp = st.session_state.df_imputed
        num_cols = df_imp.select_dtypes(include="number").columns.tolist()
        cat_cols = (df_imp.select_dtypes(exclude="number")
                          .drop(columns=["classification"], errors="ignore")
                          .columns.tolist())

        num_pipe = Pipeline([
            ("strip",   StripThousands()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ])
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        prep_pipeline = ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", ohe,      cat_cols)
        ])
        st.session_state.prep_pipeline = prep_pipeline
        st.success("📦 prep_pipeline enregistré dans la session.")
        st.markdown(
            "*Pourquoi ?* → garantir un pré-traitement identique en "
            "validation croisée **et** en production."
        )

# ╭────────── EXPANDER 4 : RÉCAP ──────────╮
with st.expander("✅ Étape 4 – Récapitulatif & stockage", expanded=False):
    if "df_imputed" in st.session_state:
        df_imp = st.session_state.df_imputed
        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes", df_imp.shape[0])
        col2.metric("Colonnes", df_imp.shape[1])
        col3.metric("NA restantes", int(df_imp.isna().sum().sum()))
        st.dataframe(df_imp.head())

        # Stockage pour les autres pages
        st.session_state.cleaned_df = df_imp

        # ─── Bouton de téléchargement ───────────────────────────
        csv = df_imp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Télécharger les données nettoyées",
            data=csv,
            file_name="donnees_nettoyees.csv",
            mime="text/csv"
        )

        st.success(
            "Les données nettoyées **et** le pipeline sont prêts pour la page "
            "« Modélisation ». Tout est conservé en mémoire."
        )
    else:
        st.info("Terminez les étapes précédentes pour accéder au résumé.")
