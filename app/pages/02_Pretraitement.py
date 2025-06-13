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


# === Step 1: Résumé des valeurs manquantes (affiché automatiquement)
st.header("🧩 Étape 1 : Résumé des valeurs manquantes")

missing_df = pd.DataFrame(df.isnull().sum(), columns=["Total NA"])
missing_df["% NA"] = (df.isnull().mean() * 100).round(1)
missing_df = missing_df[missing_df["Total NA"] > 0].sort_values("% NA", ascending=False)

if not missing_df.empty:
    st.dataframe(missing_df)
    st.session_state.missing_df = missing_df
else:
    st.success("✅ Aucune valeur manquante détectée dans le jeu de données.")

# === Step 2: Imputation des valeurs manquantes
st.header("🛠️ Étape 2 : Imputation des valeurs manquantes")

def impute_missing(df):
    df = df.copy()

    # 🔁 Step 1: Replace all '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # 🔧 Step 2: Force correct data types for numeric columns
    numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 🧹 Step 3: Normalize binary categorical columns (yes/no)
    binary_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace({
                'yes': 1, 'no': 0, 'present': 1, 'notpresent': 0,
                'abnormal': 1, 'normal': 0, 'good': 1, 'poor': 0,
                'nan': np.nan  # keep actual missing values as NaN
            })

    # 🧩 Step 4: Split into low and high missing columns
    missing_pct = df.isnull().mean() * 100
    low_missing_cols = missing_pct[missing_pct < 5].index.tolist()
    high_missing_cols = missing_pct[missing_pct >= 5].index.tolist()

    # ✅ Step 5: Impute low-missing columns
    for col in low_missing_cols:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # ⚠️ Step 6: Handle high-missing columns
    for col in high_missing_cols:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            # Instead of mode (can bias), mark as a new category
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col].fillna("missing", inplace=True)

    # 🎯 Step 7: Clean target column
    if "classification" in df.columns:
        df["classification"] = df["classification"].astype(str).str.strip().str.lower()
        df["classification"] = df["classification"].replace({
            'ckd': 1,
            'notckd': 0,
            'ckd\t': 1,  # cleaning noise
        })

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
        # Créer une copie pour l'affichage uniquement
        target_display = df_clean["classification"].replace({1: "Malade", 0: "Sain"})
        # Afficher un tableau lisible
        st.dataframe(target_display.value_counts().to_frame("Nombre de patients"))


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



