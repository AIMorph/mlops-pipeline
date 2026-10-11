import pandas as pd


def _validate_input(df, required_columns):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df is empty")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def fill_missing_values(df, numeric_cols, categorical_cols):
    _validate_input(df, list(numeric_cols) + list(categorical_cols))
    out = df.copy()
    for col in numeric_cols:
        out[col] = out[col].fillna(out[col].median())
    for col in categorical_cols:
        out[col] = out[col].fillna("Unknown")
    return out


def encode_categoricals(df, categorical_cols):
    _validate_input(df, list(categorical_cols))
    return pd.get_dummies(df.copy(), columns=list(categorical_cols), dtype=int)


def preprocess(df, numeric_cols, categorical_cols):
    filled = fill_missing_values(df, numeric_cols, categorical_cols)
    return encode_categoricals(filled, categorical_cols)