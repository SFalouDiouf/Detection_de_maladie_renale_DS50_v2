# Reproduce the full interactive preprocessing logic
# Build ColumnTransformer for 03_Modelisation.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# Pre-defined column lists from the interactive page
NUMERIC_FORCE = [
    "age", "bp", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc"
]

BIN_COLS = [
    "rbc", "pc", "pcc", "ba", "htn", "dm",
    "cad", "appet", "pe", "ane"
]

BIN_MAP = {
    "yes": 1, "no": 0,
    "present": 1, "notpresent": 0,
    "abnormal": 1, "normal": 0,
    "good": 1, "poor": 0
}

# ─────────────────────────────────────────────
# 1) Vectorized cleaning function (for FunctionTransformer)
# ─────────────────────────────────────────────
def _numeric_binary_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Remove thousand separators ',' and strip whitespaces
    - Convert NUMERIC_FORCE columns to numeric dtype
    - Map BIN_COLS columns to binary 0/1
    Returns cleaned DataFrame with the same column names.
    """
    dfc = df.copy()

    # Generic string cleaning: remove ',' and strip
    for col in dfc.columns:
        if dfc[col].dtype == object:
            dfc[col] = (dfc[col].astype(str)
                                   .str.replace(",", "", regex=False)
                                   .str.strip()
                                   .replace("nan", np.nan))

    # Force numeric columns to float
    for col in NUMERIC_FORCE:
        if col in dfc.columns:
            dfc[col] = pd.to_numeric(dfc[col], errors="coerce")

    # Map binary columns to 0/1
    for col in BIN_COLS:
        if col in dfc.columns:
            dfc[col] = (dfc[col].astype(str)
                                   .str.strip()
                                   .str.lower()
                                   .map(BIN_MAP)
                                   .replace("nan", np.nan))

    return dfc


# ─────────────────────────────────────────────
# 2) Main interface: build_preprocessor()
# ─────────────────────────────────────────────
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Given a raw input DataFrame X, build a ColumnTransformer replicating 
    the full preprocessing logic from the interactive cleaning page:
        • Numeric: _numeric_binary_clean + median imputation + scaling
        • Categorical: mode imputation + One-Hot encoding
    Returns a ready-to-use sklearn preprocessing pipeline.
    """
    cols_exist = X.columns.tolist()

    # Determine numeric columns:
    # - forced NUMERIC_FORCE + BIN_COLS (treated as numeric after mapping)
    # - any remaining numeric columns not yet included
    num_cols = [c for c in NUMERIC_FORCE + BIN_COLS if c in cols_exist]
    num_cols += [c for c in X.select_dtypes("number").columns if c not in num_cols]
    num_cols = list(dict.fromkeys(num_cols))  # deduplicate while preserving order

    # Remaining columns treated as categorical
    cat_cols = [c for c in cols_exist if c not in num_cols]

    # Full numeric pipeline: cleaning → imputation → scaling
    numeric_pipeline = Pipeline([
        ("clean", FunctionTransformer(_numeric_binary_clean,
                                      validate=False,
                                      feature_names_out="one-to-one")),
        ("imp",   SimpleImputer(strategy="median")),
        ("sc",    StandardScaler())
    ])

    # Categorical pipeline: mode imputation → One-Hot encoding
    categorical_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocess = ColumnTransformer(
        remainder="drop",
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )
    return preprocess
