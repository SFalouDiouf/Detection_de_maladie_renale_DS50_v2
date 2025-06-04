# # --- bootstrap PYTHONPATH ------------------------------
# import sys, pathlib
# ROOT = pathlib.Path(__file__).resolve().parents[2]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))
# # -------------------------------------------------------

# import streamlit as st
# from app.components import sidebar, footer
# from src import model
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier
# import plotly.express as px
# import plotly.graph_objects as go

# st.set_page_config(page_title="Modélisation", page_icon="🤖", layout="wide")
# sidebar.render()
# st.title("Étape 2 – Modélisation et validation croisée")

# # ---------- CHECK DATA ---------------------------------
# if "clean_df" not in st.session_state:
#     st.warning("Veuillez d’abord terminer l’étape d’exploration.")
#     footer.render()
#     st.stop()

# df = st.session_state["clean_df"]

# # ---------- PARAMÈTRES --------------------------------
# st.sidebar.markdown("### Paramètres d'entraînement")
# test_size = st.sidebar.slider("Taille du test", 0.1, 0.4, 0.2, 0.05, key="test_size_slider")
# target_col = st.sidebar.selectbox("Variable cible", df.columns[::-1], key="target_col_selector")
# n_folds = st.sidebar.radio("Nombre de folds CV", [5, 10], index=0, key="folds_selector")

# # ---------- TRAIN/TEST SPLIT --------------------------
# X = df.drop(columns=[target_col])
# y = df[target_col].astype(str).str.strip().str.lower()

# # (optionnel) conversion binaire
# if sorted(y.unique()) == ["ckd", "notckd"]:
#     y = y.map({"notckd": 0, "ckd": 1})

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=test_size, random_state=42, stratify=y
# )

# # ---------- OPTIMISATION RANDOM FOREST ---------------
# st.markdown("### ⚙️ Optimiser RandomForest")
# if st.button("⚙️ Optimiser RandomForest"):
#     preproc = ColumnTransformer([
#         ("num", StandardScaler(), X_train.select_dtypes(include="number").columns.tolist()),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), X_train.select_dtypes(exclude="number").columns.tolist())
#     ])
#     pipe = Pipeline([("prep", preproc), ("clf", RandomForestClassifier())])
#     param_grid = {
#         "clf__n_estimators": [50, 100, 200],
#         "clf__max_depth": [None, 5, 10]
#     }
#     best_model, best_params = model.optimize_model(pipe, param_grid, X_train, y_train)
#     st.success("✅ Optimisation terminée")
#     st.write("🔧 Meilleurs hyperparamètres :", best_params)
#     st.session_state["best_model"] = best_model

# # ---------- COMPARAISON DE MODÈLES -------------------
# if st.button("🚀 Lancer la comparaison de modèles"):
#     with st.spinner("Entraînement en cours…"):
#         cv_res = model.compare_models_with_cv(X_train, y_train, cv_splits=n_folds)

#     st.success("Comparaison terminée")
#     st.subheader("Scores moyens (validation croisée)")

#     metric_choice = st.selectbox(
#         "Afficher la métrique",
#         ["ROC AUC", "Accuracy", "Precision", "Recall", "F1"],
#         index=0
#     )

#     fig = px.bar(
#         cv_res.sort_values(metric_choice),
#         x="modèle",
#         y=metric_choice,
#         title=f"Comparaison des modèles – {metric_choice}",
#         text_auto=".2f"
#     )
#     st.plotly_chart(fig, use_container_width=True)
#     st.dataframe(cv_res.style.format(precision=3))

#     # ---------- EXPORT CSV ------------------------
#     csv_data = cv_res.drop(columns=["pipeline"]).to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "📥 Télécharger les résultats en CSV",
#         data=csv_data,
#         file_name="resultats_cv.csv",
#         mime="text/csv"
#     )

#     # ---------- MEILLEUR MODÈLE -------------------
#     best_pipe = model.select_best_model(cv_res, metric=metric_choice)
#     best_pipe.fit(X_train, y_train)
#     st.success(f"Meilleur modèle sélectionné ({metric_choice}) entraîné.")

#     st.session_state["best_model"] = best_pipe
#     st.session_state["X_test"] = X_test
#     st.session_state["y_test"] = y_test

#     # ---------- RADAR PLOT ------------------------
#     radar_df = cv_res.copy().set_index("modèle").drop(columns=["pipeline"])
#     radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

#     categories = radar_df.columns.tolist()
#     radar_fig = go.Figure()

#     for model_name in radar_df.index:
#         radar_fig.add_trace(go.Scatterpolar(
#             r=radar_df.loc[model_name].values,
#             theta=categories,
#             fill='toself',
#             name=model_name
#         ))

#     radar_fig.update_layout(
#         polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
#         showlegend=True,
#         title="Comparaison des modèles – Radar Plot"
#     )
#     st.plotly_chart(radar_fig, use_container_width=True)

# footer.render()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Modèles de Classification", layout="wide")

st.title("🏋️ Entraînement et Évaluation des Modèles de Classification")

# Vérifier que la donnée nettoyée est chargée
if 'cleaned_df' not in st.session_state:
    st.warning("⚠️ Veuillez d'abord nettoyer les données dans la page de prétraitement.")
    st.stop()

df = st.session_state.cleaned_df.copy()

# Séparer X et y
target_col = 'classification'
if target_col not in df.columns:
    st.error(f"Colonne cible '{target_col}' non trouvée dans le dataset.")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# Encodage de y si ce n’est pas numérique
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Affichage rapide des données
st.write(f"Jeu de données : {X.shape[0]} échantillons, {X.shape[1]} features.")
st.write("Colonnes features :", list(X.columns))

# Modèles à tester
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Fonction d’évaluation cross-validation stratifiée
def evaluate_model_cv(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = {
        'accuracy': cross_val_score(model, X, y, cv=skf, scoring='accuracy').mean(),
        'precision': cross_val_score(model, X, y, cv=skf, scoring='precision').mean(),
        'recall': cross_val_score(model, X, y, cv=skf, scoring='recall').mean(),
        'f1': cross_val_score(model, X, y, cv=skf, scoring='f1').mean(),
        'roc_auc': cross_val_score(model, X, y, cv=skf, scoring='roc_auc').mean(),
    }
    return scores

# Bouton pour lancer l'entraînement
if st.button("🚀 Lancer l'entraînement et l'évaluation des modèles"):
    st.info("Début de l'entraînement et de l'évaluation...")

    results = {}
    for name, model in models.items():
        st.write(f"### {name}")
        scores = evaluate_model_cv(model, X, y)
        results[name] = scores
        st.write(f"- Accuracy : {scores['accuracy']:.3f}")
        st.write(f"- Precision : {scores['precision']:.3f}")
        st.write(f"- Recall : {scores['recall']:.3f}")
        st.write(f"- F1-score : {scores['f1']:.3f}")
        st.write(f"- ROC-AUC : {scores['roc_auc']:.3f}")
        st.write("---")

    # Affichage résumé comparatif
    st.write("## 📊 Résumé comparatif des modèles")
    df_results = pd.DataFrame(results).T
    st.dataframe(df_results.style.highlight_max(axis=0))

    # Stocker les résultats en session_state pour étapes futures
    st.session_state.model_results = df_results
