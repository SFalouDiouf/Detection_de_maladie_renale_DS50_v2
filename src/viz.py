
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_missing(df: pd.DataFrame):
    """Barplot du pourcentage de valeurs manquantes par colonne."""
    missing = df.isnull().mean() * 100
    fig, ax = plt.subplots()
    missing.sort_values(ascending=False).plot.bar(ax=ax)
    ax.set_ylabel("% de valeurs manquantes")
    st.pyplot(fig)

def plot_distributions(df: pd.DataFrame):
    """Histogrammes pour chaque variable numérique."""
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution de {col}")
        st.pyplot(fig)

def plot_correlation(df: pd.DataFrame):
    """Heatmap de la matrice de corrélation."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=False, cmap='viridis', ax=ax)
    ax.set_title("Matrice de corrélation")
    st.pyplot(fig)

def plot_confusion_matrix(model, X_test, y_test):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title("Matrice de confusion")
    st.pyplot(fig)
