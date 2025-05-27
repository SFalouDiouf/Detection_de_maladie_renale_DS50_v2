"""
src/model.py
Module regroupant :
    * construction de pipelines (prétraitement + modèle)
    * comparaison par validation croisée
    * sélection et évaluation du meilleur pipeline
Aucune logique métier n’est modifiée ; on gère juste correctement
les colonnes catégorielles et les métriques multi-classes.
"""
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


# ------------------------------------------------------------------
# 1. Modèles de base à comparer
# ------------------------------------------------------------------
def _default_models():
    """Renvoie un dictionnaire {nom: estimateur}."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    return {
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
    }


# ------------------------------------------------------------------
# 2. Validation croisée + résultats
# ------------------------------------------------------------------
def compare_models_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare plusieurs modèles via validation croisée stratifiée.
    - Pipeline : StandardScaler + OneHotEncoder + estimateur
    - Si la cible est multiclasse, on retire roc_auc (source d'erreurs
      quand un fold ne contient pas toutes les classes).
    """
    models = _default_models()
    results = []

    # --- Préprocessing commun ------------------------------------------------
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    preproc = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )

    # --- Métriques -----------------------------------------------------------
    n_classes = y.nunique()
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision" if n_classes == 2 else "precision_weighted",
        "recall": "recall" if n_classes == 2 else "recall_weighted",
        "f1": "f1" if n_classes == 2 else "f1_weighted",
    }
    # On ne met l'AUC *que* pour un problème binaire
    if n_classes == 2:
        scoring["roc_auc"] = "roc_auc"

    cv = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state
    )

    # --- Boucle sur les estimateurs -----------------------------------------
    for name, clf in models.items():
        pipe = Pipeline([("prep", preproc), ("clf", clf)])

        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )

        row = {"modèle": name, "pipeline": pipe}
        row.update({metric: np.nanmean(val) for metric, val in scores.items()})
        results.append(row)

    return pd.DataFrame(results)

# ------------------------------------------------------------------
# 3. Sélection du meilleur pipeline
# ------------------------------------------------------------------
def select_best_model(cv_results: pd.DataFrame, metric: str = "test_roc_auc"):
    """Retourne le pipeline associé à la meilleure valeur de `metric`."""
    idx = cv_results[metric].idxmax()
    return cv_results.loc[idx, "pipeline"]


# ------------------------------------------------------------------
# 4. Évaluation finale sur le jeu de test
# ------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> pd.DataFrame:
    """Calcule les métriques principales sur le jeu de test."""
    preds = model.predict(X_test)
    proba_available = hasattr(model, "predict_proba")

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(
            y_test, preds, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, preds, average="weighted", zero_division=0
        ),
        "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
    }

    # ROC AUC uniquement si proba dispo et problème binaire
    if proba_available and y_test.nunique() == 2:
        proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, proba)

    return pd.DataFrame(metrics, index=["score"]).T
