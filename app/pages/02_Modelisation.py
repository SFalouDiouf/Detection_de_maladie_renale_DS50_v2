# --- bootstrap PYTHONPATH ------------------------------
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------

import streamlit as st
from app.components import sidebar, footer
from src import model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Modélisation", page_icon="🤖", layout="wide")
sidebar.render()
st.title("Étape 2 – Modélisation et validation croisée")

# ---------- CHECK DATA ---------------------------------
if "clean_df" not in st.session_state:
    st.warning("Veuillez d’abord terminer l’étape d’exploration.")
    footer.render()
    st.stop()

df = st.session_state["clean_df"]

# ---------- PARAMÈTRES --------------------------------
st.sidebar.markdown("### Paramètres d'entraînement")
test_size = st.sidebar.slider("Taille du test", 0.1, 0.4, 0.2, 0.05)
target_col = st.sidebar.selectbox("Variable cible", df.columns[::-1])
n_folds = st.sidebar.radio("Nombre de folds CV", [5, 10], index=0)

# ---------- TRAIN/TEST SPLIT AVEC CLEANING -------------
X = df.drop(columns=[target_col])
y = df[target_col].astype(str).str.strip().str.lower()

# (optionnel) convertir en classes numériques si binaire
if sorted(y.unique()) == ["ckd", "notckd"]:
    y = y.map({"notckd": 0, "ckd": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# ---------- BOUTON OPTIMISATION RANDOMFOREST ----------
st.markdown("### ⚙️ Optimiser RandomForest")

if st.button("⚙️ Optimiser RandomForest"):
    preproc = ColumnTransformer([
        ("num", StandardScaler(), X_train.select_dtypes(include="number").columns.tolist()),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X_train.select_dtypes(exclude="number").columns.tolist())
    ])

    pipe = Pipeline([("prep", preproc), ("clf", RandomForestClassifier())])

    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 5, 10]
    }

    from src.model import optimize_model
    best_model, best_params = optimize_model(pipe, param_grid, X_train, y_train)

    st.success("✅ Optimisation terminée")
    st.write("🔧 Meilleurs hyperparamètres :", best_params)
    st.session_state["best_model"] = best_model

# ---------- BOUTON COMPARAISON DE MODÈLES -------------
if st.button("🚀 Lancer la comparaison de modèles"):
    with st.spinner("Entraînement en cours…"):
        cv_res = model.compare_models_with_cv(X_train, y_train, cv_splits=n_folds)
# ---------- CHECK DATA ---------------------------------
if "clean_df" not in st.session_state:
    st.warning("Veuillez d’abord terminer l’étape d’exploration.")
    footer.render()
    st.stop()

df = st.session_state["clean_df"].copy()  # Create a copy to avoid modifying the original DataFrame

# ---------- PARAMÈTRES --------------------------------
st.sidebar.markdown("### Paramètres d'entraînement")
test_size = st.sidebar.slider("Taille du test", 0.1, 0.4, 0.2, 0.05)
target_col = st.sidebar.selectbox("Variable cible", df.columns[::-1])
n_folds = st.sidebar.radio("Nombre de folds CV", [5, 10], index=0)

# ---------- TRAIN/TEST SPLIT AVEC CLEANING -------------
X = df.drop(columns=[target_col])
y = df[target_col].astype(str).str.strip().str.lower()

# (optionnel) convertir en classes numériques si binaire
if sorted(y.unique()) == ["ckd", "notckd"]:
    y = y.map({"notckd": 0, "ckd": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# ---------- BOUTON OPTIMISATION RANDOMFOREST ----------
st.markdown("### ⚙️ Optimiser RandomForest")

if st.button("⚙️ Optimiser RandomForest"):
    preproc = ColumnTransformer([
        ("num", StandardScaler(), X_train.select_dtypes(include="number").columns.tolist()),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X_train.select_dtypes(exclude="number").columns.tolist())
    ])

    pipe = Pipeline([("prep", preproc), ("clf", RandomForestClassifier())])

    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 5, 10]
    }

    from src.model import optimize_model
    best_model, best_params = optimize_model(pipe, param_grid, X_train, y_train)

    st.success("✅ Optimisation terminée")
    st.write("🔧 Meilleurs hyperparamètres :", best_params)
    st.session_state["best_model"] = best_model

# ---------- BOUTON COMPARAISON DE MODÈLES -------------
if st.button("🚀 Lancer la comparaison de modèles"):
    with st.spinner("Entraînement en cours…"):
        cv_res = model.compare_models_with_cv(X_train, y_train, cv_splits=n_folds)

    st.success("Comparaison terminée")

    # ---------- RÉSULTATS ------------------------------
    st.subheader("Scores moyens (validation croisée)")

    metric_choice = st.selectbox(
        "Afficher la métrique",
        ["ROC AUC", "Accuracy", "Precision", "Recall", "F1"],
        index=0
    )

    fig = px.bar(
        cv_res.sort_values(metric_choice),
        x="modèle",
        y=metric_choice,
        title=f"Comparaison des modèles – {metric_choice}",
        text_auto=".2f"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cv_res.style.format(precision=3))

    # ---------- EXPORT CSV ------------------------
    csv_data = cv_res.drop(columns=["pipeline"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger les résultats en CSV",
        data=csv_data,
        file_name="resultats_cv.csv",
        mime="text/csv"
    )

    # ---------- MEILLEUR MODÈLE ------------------------
    best_pipe = model.select_best_model(cv_res, metric=metric_choice)
    best_pipe.fit(X_train, y_train)

    st.success(f"Meilleur modèle sélectionné ({metric_choice}) entraîné.")
    st.session_state["best_model"] = best_pipe
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

    # ---------- RADAR PLOT ------------------------
    radar_df = cv_res.copy()
    radar_df.set_index("modèle", inplace=True)
    radar_df = radar_df.drop(columns=["pipeline"])
    radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

    categories = radar_df.columns.tolist()
    radar_fig = go.Figure()

    for model_name in radar_df.index:
        radar_fig.add_trace(go.Scatterpolar(
            r=radar_df.loc[model_name].values,
            theta=categories,
            fill='toself',
            name=model_name
        ))

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Comparaison des modèles – Radar Plot"
    )

    st.plotly_chart(radar_fig, use_container_width=True)

    st.success("Comparaison terminée")

    # ---------- RÉSULTATS ------------------------------
    st.subheader("Scores moyens (validation croisée)")

    metric_choice = st.selectbox(
        "Afficher la métrique",
        ["ROC AUC", "Accuracy", "Precision", "Recall", "F1"],
        index=0
    )

    fig = px.bar(
        cv_res.sort_values(metric_choice),
        x="modèle",
        y=metric_choice,
        title=f"Comparaison des modèles – {metric_choice}",
        text_auto=".2f"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cv_res.style.format(precision=3))

    # ---------- EXPORT CSV ------------------------
    csv_data = cv_res.drop(columns=["pipeline"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger les résultats en CSV",
        data=csv_data,
        file_name="resultats_cv.csv",
        mime="text/csv"
    )

    # ---------- MEILLEUR MODÈLE ------------------------
    best_pipe = model.select_best_model(cv_res, metric=metric_choice)
    best_pipe.fit(X_train, y_train)

    st.success(f"Meilleur modèle sélectionné ({metric_choice}) entraîné.")
    st.session_state["best_model"] = best_pipe
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

    # ---------- RADAR PLOT ------------------------
    radar_df = cv_res.copy()
    radar_df.set_index("modèle", inplace=True)
    radar_df = radar_df.drop(columns=["pipeline"])
    radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

    categories = radar_df.columns.tolist()
    radar_fig = go.Figure()

    for model_name in radar_df.index:
        radar_fig.add_trace(go.Scatterpolar(
            r=radar_df.loc[model_name].values,
            theta=categories,
            fill='toself',
            name=model_name
        ))

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Comparaison des modèles – Radar Plot"
    )

    st.plotly_chart(radar_fig, use_container_width=True)

footer.render()