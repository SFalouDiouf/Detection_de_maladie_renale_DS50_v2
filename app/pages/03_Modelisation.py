# ─────────────────────────────────────────────
# 03_Modelisation.py — v3 CKD-ready
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd, numpy as np
from pathlib import Path
import joblib, datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, PrecisionRecallDisplay,
                             roc_auc_score, precision_recall_curve)

from sklearn.pipeline   import Pipeline
from sklearn.compose    import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute     import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.svm               import SVC
from lightgbm                  import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from app.pages.quick_clean import build_preprocessor
import re  
sns.set_style("whitegrid")          # rendu plus doux

# ╭── CONFIG STREAMLIT ─────────────────────────────────────────╮
st.set_page_config(page_title="🤖 Modélisation", page_icon="🧠", layout="wide")
st.title("🤖 Étape 3 — Modélisation CKD (v3)")

if "raw_df" not in st.session_state:
    st.warning("Importez d’abord le CSV (pages « Exploration » ou "
               "« Pré-traitement »).")
    st.stop()
df = st.session_state.raw_df.copy()
df["classification"] = (
    df["classification"]
      .astype(str)
      .str.strip()
      .str.lower()
      .apply(lambda s: re.sub(r"\s+", "", s))
      .replace({"ckd": 1, "notckd": 0})
      .astype(int)
)
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)
y  = df.pop("classification")
X  = df

# ╭── SPLIT HOLD-OUT RÉGLABLE ──────────────────────────────────╮
st.subheader("📦 Séparation *external* test")
test_pct = st.slider("Pourcentage du hold-out", 10, 40, 20, 5)
X_dev, X_hold, y_dev, y_hold = train_test_split(
    X, y, stratify=y, test_size=test_pct / 100, random_state=42
)
st.write(f"Hold-out : {X_hold.shape}")

# ╭── PIPELINE DE PRÉ-TRAITEMENT ───────────────────────────────╮
prep = build_preprocessor(X_dev) 

# ╭── CANDIDATS MODÈLES ────────────────────────────────────────╮
models = {
    "Logistic R": LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                            random_state=42),
    "SVM-RBF": SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LightGBM": LGBMClassifier(
        n_estimators=400, learning_rate=0.05,
        class_weight="balanced", random_state=42,
        n_jobs=-1, verbosity=-1, verbose=-1   # ← coupe les warnings
    )
}

pipelines = {name: Pipeline([("prep", prep), ("clf", clf)])
             for name, clf in models.items()}

scoring = {"ROC_AUC": "roc_auc",
           "AP":       "average_precision",
           "RECALL":   "recall",
           "PREC":     "precision"}

# ╭── COMPARAISON PAR VALIDATION CROISÉE ───────────────────────╮
if st.button("🚀 Comparer les modèles"):
    min_per_class = y_dev.value_counts().min()
    safe_cv = max(2, min(5, min_per_class))
    cv = StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=42)
    res = {}
    with st.spinner("Cross-validation…"):
        for name, pipe in pipelines.items():
            scores = cross_validate(pipe, X_dev, y_dev,
                                    scoring=scoring, cv=cv, n_jobs=-1)
            res[name] = {m: np.mean(scores[f"test_{m}"]) for m in scoring}

    res_df = (pd.DataFrame(res).T
              .round(3)
              .sort_values("ROC_AUC", ascending=False))
    st.dataframe(res_df)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(res_df.T, annot=True, cmap="coolwarm",
                linewidths=.5, ax=ax)
    ax.set_title("CV 5-fold — moyennes")
    st.pyplot(fig)

    best_name = res_df.index[0]
    st.success(f"🏆 Meilleur modèle CV : **{best_name}**")
    best_pipe = pipelines[best_name]

    # ╭── CALIBRATION & ENTRAÎNEMENT COMPLET DEV ───────────────╮
    st.subheader("📏 Calibration isotone + fit complet")
    calib = CalibratedClassifierCV(best_pipe, method="isotonic", cv=safe_cv)
    calib.fit(X_dev, y_dev)

    # ╭── SEUIL OPTIMAL (F1) ───────────────────────────────────╮
    proba_dev = calib.predict_proba(X_dev)[:, 1]
    prec, rec, thr = precision_recall_curve(y_dev, proba_dev)
    f1      = 2 * prec * rec / (prec + rec + 1e-8)
    opt_thr = thr[np.argmax(f1)]
    st.write(f"Seuil optimal F1 : **{opt_thr:.2f}** (F1 = {f1.max():.3f})")

    # ╭── ÉVALUATION HOLD-OUT ──────────────────────────────────╮
    st.subheader("🧪 Évaluation hold-out")
    proba_test = calib.predict_proba(X_hold)[:, 1]
    y_pred     = (proba_test >= opt_thr).astype(int)

    cm = confusion_matrix(y_hold, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm, colorbar=False)
    st.pyplot(fig_cm)

    st.caption(f"ROC-AUC : {roc_auc_score(y_hold, proba_test):.3f} | "
               f"PR-AUC : {np.trapz(rec, prec):.3f}")

    fig_roc, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_hold, proba_test, ax=ax)
    st.pyplot(fig_roc)

    fig_pr, ax2 = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_hold, proba_test, ax=ax2)
    ax2.axhline(y_hold.mean(), ls="--", color="grey")
    st.pyplot(fig_pr)

    # ╭── STOCKAGE EN SESSION POUR LA PAGE « Prédiction » ──────╮
    st.session_state.best_model      = calib
    st.session_state.best_threshold  = float(opt_thr)
    st.session_state.train_columns   = X_dev.columns.tolist()
    st.success("✅ Modèle, seuil et colonnes sauvegardés pour la page « Prédiction ».")

    # ╭── EXPORT SUR DISQUE ────────────────────────────────────╮
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = Path(f"ckd_model_v3_{best_name.replace(' ', '_')}_{ts}.joblib")
    joblib.dump(calib, path)
    with open(path, "rb") as f:
        st.download_button("💾 Télécharger le modèle calibré",
                           data=f, file_name=path.name,
                           mime="application/octet-stream")
