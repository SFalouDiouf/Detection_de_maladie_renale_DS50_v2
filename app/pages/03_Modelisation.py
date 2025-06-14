# ─────────────────────────────────────────────
# 03_Modelisation.py — v3 CKD-ready
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd, numpy as np
from pathlib import Path
import joblib, shap, warnings, datetime

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, RandomizedSearchCV)
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, PrecisionRecallDisplay,
                             roc_auc_score, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ╭────────── CONFIG ──────────╮
st.set_page_config(page_title="🤖 Modélisation", page_icon="🧠", layout="wide")
st.title("🤖 Étape 3 – Modélisation CKD (v3)")

# ╭────────── DATA CHECK ──────╮
if "cleaned_df" not in st.session_state:
    st.warning("Passez d’abord par « Pré-traitement ».")
    st.stop()

df = st.session_state.cleaned_df.copy()
y = df.pop("classification")
X = df

# ╭────────── HOLD-OUT FINAL ──╮
st.subheader("📦 Split *external* test (20 %)")
X_dev, X_hold, y_dev, y_hold = train_test_split(
    X, y, stratify=y, test_size=0.20, random_state=42
)
st.write("Hold-out conservé pour la fin : ", X_hold.shape)

# ╭────────── PREPROCESS PIPE ─╮
num_cols = X_dev.select_dtypes("number").columns.tolist()
cat_cols = X_dev.select_dtypes(exclude="number").columns.tolist()

prep = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler())
    ]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# ╭────────── MODEL LIST ──────╮
models = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Logistic R": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42),
    "SVM-RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LightGBM": LGBMClassifier(
        n_estimators=400, learning_rate=0.05,
        class_weight="balanced", random_state=42
    )
}
pipelines = {n: Pipeline([("prep", prep), ("clf", m)]) for n, m in models.items()}

scoring = {
    "ROC_AUC": "roc_auc",
    "AP": "average_precision",   # PR-AUC
    "RECALL": "recall",
    "PREC": "precision"
}

# ╭────────── CV COMPARISON ───╮
if st.button("🚀 Comparer les modèles"):
    cv   = StratifiedKFold(5, shuffle=True, random_state=42)
    res  = {}
    with st.spinner("Cross-validation…"):
        for n, p in pipelines.items():
            scr = cross_validate(p, X_dev, y_dev, scoring=scoring, cv=cv, n_jobs=-1)
            res[n] = {k: np.mean(scr[f"test_{k}"]) for k in scoring}
    res_df = pd.DataFrame(res).T.round(3).sort_values("ROC_AUC", ascending=False)
    st.dataframe(res_df)

    # heatmap
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(res_df.T, annot=True, cmap="coolwarm", ax=ax, linewidths=.5)
    ax.set_title("CV 5-fold – moyennes")
    st.pyplot(fig)

    best_name = res_df.index[0]
    st.success(f"🏆 Sélection : **{best_name}**")
    best_pipe = pipelines[best_name]

    # ╭────────── Calibration ────╮
    st.subheader("📏 Calibration isotone + fit full DEV")
    calib = CalibratedClassifierCV(best_pipe, method="isotonic", cv=5)
    calib.fit(X_dev, y_dev)

    # ╭────────── Threshold opt. ─╮
    proba_dev = calib.predict_proba(X_dev)[:, 1]
    prec, rec, th = precision_recall_curve(y_dev, proba_dev)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    opt_thr = th[np.argmax(f1)]
    st.write(f"Seuil F1‐max : **{opt_thr:.2f}** (F1={f1.max():.3f})")

    # ╭────────── TEST FINAL ─────╮
    st.subheader("🧪 Évaluation hold-out (jamais vu)")
    proba_test = calib.predict_proba(X_hold)[:, 1]
    y_pred = (proba_test >= opt_thr).astype(int)

    cm = confusion_matrix(y_hold, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm, colorbar=False)
    st.pyplot(fig_cm)

    st.caption(
        f"ROC-AUC : {roc_auc_score(y_hold, proba_test):.3f} | "
        f"Precision-Recall AUC : {np.trapz(rec, prec):.3f}"
    )

    # ╭────────── ROC / PR CURVE ─╮
    fig_roc, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_hold, proba_test, ax=ax)
    st.pyplot(fig_roc)
    fig_pr, ax2 = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_hold, proba_test, ax=ax2)
    ax2.axhline(y_hold.mean(), ls="--", color="grey")  # baseline
    st.pyplot(fig_pr)

    # ╭────────── SHAP GLOBAL ────╮
    st.subheader("🔎 Interprétation SHAP (LightGBM / RF / LR)")
    try:
        expl = shap.Explainer(calib.base_estimator_["clf"])
        shap_vals = expl(calib.base_estimator_["prep"].transform(X_hold))
        fig_shap = shap.summary_plot(shap_vals, show=False, plot_size=(8, 4))
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.info(f"SHAP indisponible : {e}")

    # ╭────────── EXPORT JOBLIB ──╮
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = Path(f"ckd_model_{ts}.joblib")
    joblib.dump(calib, path)
    with open(path, "rb") as f:
        st.download_button("💾 Télécharger le modèle calibré",
                           data=f, file_name=path.name, mime="application/octet-stream")
