import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Pipeline de Prétraitement", layout="wide")
st.title("🧼 Pipeline de Prétraitement des Données")

# === INIT
if 'uploaded_df' not in st.session_state:
    st.warning("Veuillez d'abord charger un fichier dans la page principale.")
    st.stop()

df = st.session_state.uploaded_df.copy()

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

df["classification"] = df["classification"].astype(str).str.strip().str.lower()


# === Step 1: Résumé des valeurs manquantes
st.header("🧩 Étape 1 : Résumé des valeurs manquantes")
if st.button("Afficher le résumé des valeurs manquantes"):
    missing_df = pd.DataFrame(df.isnull().sum(), columns=["Total NA"])
    missing_df["% NA"] = (df.isnull().mean() * 100).round(1)
    missing_df = missing_df[missing_df["Total NA"] > 0].sort_values("% NA", ascending=False)
    st.dataframe(missing_df)
    st.session_state.missing_df = missing_df

# === Step 2: Imputation des valeurs manquantes
st.header("🛠️ Étape 2 : Imputation des valeurs manquantes")

def impute_missing(df):
    df = df.copy()

    # 🔁 Replace all '?' with NaN (in all columns, not just object types)
    df.replace("?", np.nan, inplace=True)

    # 🔍 Calculate missing value percentage per column
    missing_pct = df.isnull().mean() * 100
    low_missing_cols = missing_pct[missing_pct < 5].index.tolist()
    high_missing_cols = missing_pct[missing_pct >= 5].index.tolist()

    

    # ✅ Treat low-missing columns
    for col in low_missing_cols:
        if df[col].dtype in ("float64", "int64", "int32"):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # ⚠️ For high-missing columns:
    for col in high_missing_cols:
        if df[col].dtype in ("float64", "int64", "int32"):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
       

    # ✅ Normalize target column again if needed
    if "classification" in df.columns:
        df["classification"] = df["classification"].astype(str).str.strip().str.lower()

    return df



if st.button("Imputer les valeurs manquantes"):
    df_imputed = impute_missing(df)
    st.session_state.df_imputed = df_imputed
    st.success("✅ Valeurs manquantes imputées.")
    

# === Résumé visuel après imputation ===
if "df_imputed" in st.session_state:
    df_clean = st.session_state.df_imputed
    st.header("✅ Jeu de données prêt")

    st.success("Toutes les valeurs manquantes ont été traitées. Le jeu de données est propre et prêt à être utilisé pour la modélisation.")

    # 1. Vérification visuelle des NA
    na_summary = df_clean.isnull().sum()
    

    # 2. Shape + types
    st.subheader("📋 Aperçu global")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de lignes", f"{df_clean.shape[0]}")
    col2.metric("Nombre de colonnes", f"{df_clean.shape[1]}")
    col3.metric("Valeurs manquantes", f"{int(na_summary.sum())}")

    # 3. Répartition de la variable cible
    if "classification" in df_clean.columns:
        st.subheader("🎯 Répartition de la variable cible (classification)")
        st.dataframe(df_clean["classification"].value_counts().to_frame("Count"))
        st.bar_chart(df_clean["classification"].value_counts())

    # 4. Aperçu des 10 premières lignes
    st.subheader("🔍 Aperçu des données nettoyées")
    st.dataframe(df_clean.head())

# ✅ Télécharger uniquement si l'imputation a été faite
if "df_imputed" in st.session_state:
    df_clean = st.session_state.df_imputed  # <- define df_clean
    st.session_state.cleaned_df = df_clean  # <- store cleaned df for other pages

    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Télécharger les données nettoyées",
        data=csv,
        file_name="donnees_nettoyees.csv",
        mime="text/csv"
    )



