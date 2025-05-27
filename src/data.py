
import pandas as pd

import pandas as pd

def load_dataset(csv_file):
    """
    Lit un CSV uploadé avec Streamlit. Remet le curseur en début de
    fichier à chaque appel pour éviter EmptyDataError.
    """
    if hasattr(csv_file, "seek"):
        csv_file.seek(0)          # ← reset du buffer
    return pd.read_csv(csv_file)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage simple (placeholder) : imputation NA par médiane/mode."""
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include='number').columns:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in cleaned.select_dtypes(exclude='number').columns:
        cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
    return cleaned
