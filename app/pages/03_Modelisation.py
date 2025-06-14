# ─────────────────────────────────────────────
# 03_Modelisation.py — entraînement & comparaison
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
)
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt

# ╭────────── CONFIG STREAMLIT ──────────╮
st.set_page_config(page_title="🤖 Modélisation", page_icon="🧠", layout="wide")
st.title("🤖 Étape 3 – Entraînement, Évaluation et Comparaison des Modèles")

# ╭────────── CONTRÔLE PRÉ-REQUIS ───────╮
if "cleaned_df" not in st.session_state:
    st.warning("Exécutez d’abord la page « Pré-traitement ». 🚩")
    st.stop()

df = st.session_state.cleaned_df.copy()

target = "classification"
if target not in df.columns:
    st.error("La colonne cible 'classification' est manquante.")
    st.stop()

# ╭────────── SPLIT TRAIN / TEST ─────────╮
st.subheader("📦 Séparation Train / Test")
test_size = st.slider("Taille du test (%)", 10, 40, 20, 5) / 100
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=test_size, random_state=42
)
st.write(f"Shape X_train : {X_train.shape}  •  Shape X_test : {X_test.shape}")
st.write("Répartition cible (train) :", pd.Series(y_train)
         .value_counts(normalize=True).round(2))

# ╭────────── PRÉ-PROCESSUS LOCAL (sans 'classification') ──╮
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

prep = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", ohe,      cat_cols)
])

# ╭────────── DÉFINITION DES PIPELINES ────╮
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs"),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}
pipelines = {name: Pipeline([("prep", prep), ("clf", clf)])
             for name, clf in models.items()}

scoring = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall"
}

# ╭────────── ENTRAÎNEMENT COMPARATIF ─────╮
if st.button("🚀 Lancer la comparaison des modèles"):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X_train, y_train,
                                scoring=scoring, cv=skf, n_jobs=-1)
        results[name] = {m.upper(): np.mean(scores[f"test_{m}"])
                         for m in scoring}
    res_df = pd.DataFrame(results).T.round(3)
    st.dataframe(res_df)

    # Heat-map
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(res_df.T, annot=True, cmap="Blues", ax=ax, linewidths=.5)
    ax.set_title("Moyennes des métriques (CV 5-fold)")
    st.pyplot(fig)

    best_name = res_df["ROC_AUC"].idxmax()
    st.info(f"🏆 Meilleur modèle initial : **{best_name}**")
    best_pipe = pipelines[best_name]

    # ╭────────── HYPERPARAMETERS SEARCH ─────╮
    st.subheader("🔍 Recherche d’hyperparamètres (sur le meilleur)")
    if best_name == "Random Forest":
        param_grid = {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10]
        }
    elif best_name == "Logistic Regression":
        param_grid = {"clf__C": [0.01, 0.1, 1, 10],
                      "clf__penalty": ["l2"]}
    else:
        param_grid = {}

    if param_grid:
        search = RandomizedSearchCV(
            best_pipe, param_grid, n_iter=10,
            scoring="roc_auc", cv=skf, n_jobs=-1, random_state=42
        )
        with st.spinner("🔬 Recherche en cours…"):
            search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        st.write("Meilleur AUC CV :", round(search.best_score_, 3))
        st.write("Hyperparamètres :", search.best_params_)

    # ╭────────── TRAIN FINAL + TEST SET ──────╮
    best_pipe.fit(X_train, y_train)
    st.session_state.best_model = best_pipe   # pour page suivante

    y_pred  = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]

    st.subheader("🧾 Matrice de confusion")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm, colorbar=False)
    st.pyplot(fig_cm)

    st.subheader("📈 Courbes ROC & PR")
    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc)
    st.pyplot(fig_roc)

    fig_pr, ax_pr = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax_pr)
    st.pyplot(fig_pr)

    # ╭────────── TÉLÉCHARGEMENT DU MODELE ────╮
    model_path = Path("best_model.joblib")
    joblib.dump(best_pipe, model_path)
    with open(model_path, "rb") as f:
        st.download_button(
            "💾 Télécharger le modèle (joblib)",
            data=f,
            file_name="best_model.joblib"
        )
