# ─────────────────────────────────────────────
# 02_Pretraitement.py – Pipeline pas-à-pas (expanders)
# ─────────────────────────────────────────────
import sys, pathlib, warnings
import streamlit as st
import pandas as pd, numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="🧼 Pré-traitement", page_icon="🧹", layout="wide")
st.title("🧹 Pipeline de Pré-traitement des Données")

# ╭──────────────────────────────────────────╮
# │ 0. Récupération du DataFrame             │
# ╰──────────────────────────────────────────╯
if "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"].copy()
elif "clean_df" in st.session_state:
    df = st.session_state["clean_df"].copy()
else:
    st.warning("Veuillez d'abord charger et explorer le fichier.")
    st.stop()

# Nettoyages légers (aucune suppression de colonne)
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)
df.replace("?", np.nan, inplace=True)
for col in ["pcv", "wc", "rc"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
if "classification" in df.columns:
    df["classification"] = df["classification"].astype(str).str.strip().str.lower()

numeric_cols   = df.select_dtypes(include=["number"]).columns.tolist()
categoric_cols = df.select_dtypes(exclude=["number"]).drop(columns=["classification"], errors="ignore").columns.tolist()

# ╭──────────────────────────────────────────╮
# │ 1. Analyse des valeurs manquantes        │
# ╰──────────────────────────────────────────╯
with st.expander("🔍 Étape 1 – Analyse des valeurs manquantes", expanded=True):
    st.markdown(
        "Nous commençons par **quantifier** les valeurs manquantes afin de "
        "choisir une stratégie d’imputation adaptée."
    )
    na_df = (
        df.isna().mean().mul(100).round(1)
        .reset_index().rename(columns={"index": "Colonne", 0: "% NA"})
        .sort_values("% NA", ascending=False)
    )
    st.dataframe(na_df, height=260)
    st.session_state["na_df"] = na_df

# ╭──────────────────────────────────────────╮
# │ 2. Imputation                            │
# ╰──────────────────────────────────────────╯
with st.expander("🛠️ Étape 2 – Imputation", expanded=False):
    st.markdown(
        """
        **Règles fixes basées sur l'exploration :**  
        • **≤ 5 % NA** → médiane (num) ou mode (cat)  
        • **5 – 40 % NA** → KNN Imputer (k = 3) pour variables numériques  
        • **> 40 % NA** → KNN quand même (on conserve l’info mais on note l’incertitude)
        """
    )
    if st.button("🔁 Exécuter l’imputation"):
        NA_PCT = df.isna().mean().mul(100)
        num_med = [c for c in numeric_cols if NA_PCT[c] <= 5]
        num_knn = [c for c in numeric_cols if NA_PCT[c] > 5]
        cat_all = categoric_cols

        df_imp = df.copy()
        df_imp[num_med] = SimpleImputer(strategy="median").fit_transform(df_imp[num_med])
        if num_knn:
            df_imp[num_knn] = KNNImputer(n_neighbors=3).fit_transform(df_imp[num_knn])
        for c in cat_all:
            df_imp[c] = df_imp[c].fillna(df_imp[c].mode()[0])

        st.session_state["df_imp"] = df_imp
        st.success("Imputation terminée ✅")
        st.dataframe(df_imp.head())

# ╭──────────────────────────────────────────╮
# │ 3. Standardisation & encodage            │
# ╰──────────────────────────────────────────╯
with st.expander("⚙️ Étape 3 – Standardisation & One-Hot Encodage", expanded=False):
    if "df_imp" not in st.session_state:
        st.info("Veuillez d'abord lancer l’imputation.")
    else:
        df_imp = st.session_state["df_imp"]

        # OneHotEncoder : compatibilité scikit-learn ≥1.2
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("scaler", StandardScaler())
                ]), numeric_cols),
                ("cat", Pipeline([
                    ("ohe", ohe)
                ]), categoric_cols)
            ]
        )

        if st.button("⚙️ Appliquer Scale + OHE"):
            X = df_imp.drop(columns=["classification"], errors="ignore")
            X_prep = preprocessor.fit_transform(X)

            st.session_state["prep_pipeline"] = preprocessor
            st.session_state["X_prep"] = X_prep

            st.success(f"Transformation effectuée ! Shape finale : {X_prep.shape}")
            st.write(pd.DataFrame(X_prep[:5]).head())

            st.markdown(
                """
                **Pourquoi ces choix ?**  
                • `StandardScaler` aligne les échelles des variables numériques  
                • `OneHotEncoder` convertit les catégorielles, et `handle_unknown="ignore"` évite les erreurs si une nouvelle modalité apparaît en production
                """
            )

# ╭──────────────────────────────────────────╮
# │ 4. Résumé & passage à la suite           │
# ╰──────────────────────────────────────────╯
with st.expander("✅ Étape 4 – Résumé final", expanded=False):
    if "X_prep" in st.session_state:
        X_prep = st.session_state["X_prep"]
        st.metric("Observations", X_prep.shape[0])
        st.metric("Features finales", X_prep.shape[1])
        st.success("Les données préparées et le pipeline sont prêts en mémoire !")
        st.markdown(
            "👉 Ouvrez maintenant la page **Modélisation** pour entraîner le modèle ; "
            "vous y trouverez `prep_pipeline`, `X_prep` et `df_imp` déjà chargés."
        )
    else:
        st.info("Terminez l’étape 3 pour voir ce récapitulatif.")
