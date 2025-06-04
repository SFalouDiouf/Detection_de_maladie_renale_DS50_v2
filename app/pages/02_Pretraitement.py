import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Prétraitement", layout="wide")
st.title("🧹 Prétraitement des Données")

# --- Initialisation des flags dans session_state ---
if "step_1_done" not in st.session_state:
    st.session_state.step_1_done = False
if "step_2_done" not in st.session_state:
    st.session_state.step_2_done = False
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "final_df" not in st.session_state:
    st.session_state.final_df = None

# --- Chargement des données uploadées ---
if 'uploaded_df' not in st.session_state:
    st.warning("📁 Veuillez d'abord charger les données dans la page d'accueil.")
    st.stop()

df = st.session_state.uploaded_df.copy()

# === Fonctions de nettoyage ===
def clean_wc_rc(df):
    for col in ['wc', 'rc']:
        if col in df.columns:
            df[col] = df[col].replace(['?', 'normal'], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
    return df

def impute_missing_values(df):
    df = df.copy()
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'].fillna(df['age'].median(), inplace=True)

    numeric_mean_cols = ['bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv']
    for col in numeric_mean_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].mean(), inplace=True)

    df = clean_wc_rc(df)

    categorical_mode_cols = ['sg', 'al', 'su', 'pcc', 'ba']
    for col in categorical_mode_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    for col in ['dm', 'cad', 'htn', 'appet', 'pe', 'ane']:
        df[col] = df[col].replace('?', np.nan)
        df[col].fillna(df[col].mode()[0], inplace=True)

    for col in ['rbc', 'pc']:
        df[col].fillna('unknown', inplace=True)

    return df

def full_preprocessing(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        median = df[col].median()
        df[col] = np.where((df[col] < lower) | (df[col] > upper), median, df[col])

    label_encoding = {
        'normal': 0, 'abnormal': 1,
        'yes': 1, 'no': 0,
        'present': 1, 'notpresent': 0,
        'good': 1, 'poor': 0,
        'unknown': -1
    }

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].map(label_encoding).fillna(df[col])

    df = pd.get_dummies(df, drop_first=True)

    return df

# --- Bouton Étape 1 ---
st.subheader("Étape 1 – Gestion des valeurs manquantes")
if st.button("🚀 Start Cleaning") or st.session_state.step_1_done:
    if not st.session_state.step_1_done:
        cleaned_df = impute_missing_values(df)
        st.session_state.cleaned_df = cleaned_df
        st.session_state.step_1_done = True
    else:
        cleaned_df = st.session_state.cleaned_df

    st.success("✅ Valeurs manquantes remplies.")
    st.dataframe(cleaned_df.isnull().sum().to_frame("Valeurs manquantes"))

# --- Bouton Étape 2 ---
if st.session_state.step_1_done:
    st.subheader("Étape 2 – Nettoyage complet et encodage")

    if st.button("▶️ Continuer le Prétraitement") or st.session_state.step_2_done:
        if not st.session_state.step_2_done:
            progress_bar = st.progress(0)
            status = st.empty()

            time.sleep(0.5)
            status.info("🔍 Nettoyage des outliers...")
            progress_bar.progress(25)

            time.sleep(0.5)
            df2 = st.session_state.cleaned_df.copy()
            final_df = full_preprocessing(df2)
            progress_bar.progress(75)

            time.sleep(0.5)
            status.success("✅ Encodage terminé.")
            progress_bar.progress(100)

            st.session_state.final_df = final_df
            st.session_state.step_2_done = True
        else:
            final_df = st.session_state.final_df

        st.success("🎉 Prétraitement terminé :")
        st.markdown("- Outliers remplacés par la médiane")
        st.markdown("- Encodage (Label + One-Hot) appliqué")
        st.dataframe(final_df.head())

        # --- Bouton téléchargement ---
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger les données nettoyées", data=csv, file_name="donnees_nettoyees.csv", mime="text/csv")
