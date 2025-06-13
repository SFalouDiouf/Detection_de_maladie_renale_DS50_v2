import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modélisation", layout="wide")
st.title("🤖 Étape 3 – Entraînement, Évaluation et Comparaison des Modèles")

# === Vérification des données nettoyées
if "cleaned_df" not in st.session_state:
    st.warning("Veuillez d'abord effectuer le prétraitement.")
    st.stop()

df = st.session_state.cleaned_df.copy()

# === Séparation features / cible
target_col = "classification"
if target_col not in df.columns:
    st.error("La colonne cible 'classification' est manquante.")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# Encodage de la cible
if y.dtype == "object":
    y = y.str.strip().str.lower()
    le = LabelEncoder()
    y = le.fit_transform(y)
    st.session_state["label_encoder"] = le  # Sauvegarde pour plus tard


# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# === Split train/test
st.subheader("📦 Séparation Train / Test")
test_size = st.slider("Taille du test (%)", min_value=10, max_value=40, value=20, step=5) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=42)
st.session_state.X_train = X_train  # <-- Add this line


st.write(f"📊 X_train shape: {X_train.shape}")
st.write(f"📊 X_test shape: {X_test.shape}")
st.write("🎯 Répartition cible (y_train):")
st.caption("💡 0 = Sain, 1 = Malade")
st.write(pd.Series(y_train).value_counts())

from sklearn.model_selection import RandomizedSearchCV

def recherche_meilleurs_modeles(X_train, y_train):
    st.subheader("🔍 Recherche des meilleurs hyperparamètres (RF & LogisticRegression)")

    # === 1. Random Forest
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    param_rf = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }

    search_rf = RandomizedSearchCV(
        rf, param_distributions=param_rf,
        n_iter=20, scoring="roc_auc", cv=5,
        n_jobs=-1, random_state=42
    )

    # === 2. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)
    param_lr = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"]
    }

    search_lr = RandomizedSearchCV(
        lr, param_distributions=param_lr,
        n_iter=10, scoring="roc_auc", cv=5,
        n_jobs=-1, random_state=42
    )

    with st.spinner("🔬 Recherche des meilleurs modèles..."):
        search_rf.fit(X_train, y_train)
        search_lr.fit(X_train, y_train)

    # Résultats
    st.success("✅ Recherche terminée.")

    st.write("📈 **Random Forest ROC AUC** :", round(search_rf.best_score_, 3))
    st.write(" **Params RF** :", search_rf.best_params_)

    st.write("📈 **Logistic Regression ROC AUC** :", round(search_lr.best_score_, 3))
    st.write(" **Params LR** :", search_lr.best_params_)

    # Comparaison et choix du meilleur
    if search_rf.best_score_ >= search_lr.best_score_:
        best_model_name = "Random Forest"
        best_model = search_rf.best_estimator_
        best_auc = search_rf.best_score_
    else:
        best_model_name = "Logistic Regression"
        best_model = search_lr.best_estimator_
        best_auc = search_lr.best_score_

    st.info(f" Meilleur modèle : **{best_model_name}** avec AUC = {round(best_auc, 3)}")

    return best_model

# === Définition des modèles
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)  
}


# === Scoring metrics
scoring = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
}

# === Entraînement et évaluation
if st.button("🚀 Lancer l'entraînement et la comparaison des modèles"):
    st.info("En cours...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=skf)
        mean_scores = {metric.upper(): np.mean(scores[f'test_{metric}']) for metric in scoring}
        results[name] = mean_scores

    results_df = pd.DataFrame(results).T.round(3)

    st.success("✅ Modèles évalués avec succès")
    st.dataframe(results_df)

    # 🔥 Meilleur modèle
    best_model_name = results_df["ROC_AUC"].idxmax()
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    st.session_state.best_model = best_model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

    # === 🔍 Heatmap
    st.subheader("📈 Comparaison visuelle des modèles")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(results_df.T, annot=True, fmt=".3f", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("📊 Moyennes des métriques en validation croisée")
    st.pyplot(fig)

    st.subheader("📊 Comparaison des scores ROC_AUC par modèle")

    fig_bar, ax_bar = plt.subplots()
    results_df["ROC_AUC"].plot(kind="bar", color="skyblue", edgecolor="black", ax=ax_bar)
    ax_bar.set_ylabel("ROC_AUC")
    ax_bar.set_title("Score ROC_AUC pour chaque modèle")
    st.pyplot(fig_bar)



    # Appelle la fonction pour trouver le meilleur modèle
    best_model = recherche_meilleurs_modeles(X_train, y_train)

    # Entraînement final avec le meilleur modèle
    best_model.fit(X_train, y_train)

    # Sauvegarde dans session_state pour les prédictions
    st.session_state.best_model = best_model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

# === 🔮 Prédiction sur les données de test
if "best_model" in st.session_state:
    st.subheader("🔮 Prédictions sur le jeu de test avec le meilleur modèle")

    best_model = st.session_state.best_model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    y_pred = best_model.predict(X_test)

    results_df = pd.DataFrame({
        "y_test (réel)": y_test,
        "y_pred (prédit)": y_pred
    })

    st.write("Aperçu des 5 premières prédictions :")
    st.dataframe(results_df.head())

    # ✅ Matrice de confusion ici (à l'intérieur du bloc)
    st.subheader("🧾 Matrice de Confusion")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm, colorbar=False)
    st.pyplot(fig_cm)

else:
    st.info("ℹ️ Veuillez d'abord lancer l'entraînement pour effectuer des prédictions.")

