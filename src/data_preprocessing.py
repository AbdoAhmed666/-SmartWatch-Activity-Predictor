# src/data_preprocessing.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and NA rows."""
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df

def encode_categorical(df: pd.DataFrame, target_column: str, encode_target: bool = True):
    """
    Encode object/string columns into numeric using LabelEncoder.
    Returns (df_encoded, encoders_dict).
    encoders_dict maps column -> LabelEncoder
    """
    df_encoded = df.copy()
    encoders = {}
    for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
        if col == target_column:
            if encode_target:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                encoders[col] = le
            # else: skip encoding target
            continue
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    return df_encoded, encoders

def apply_encoders(df: pd.DataFrame, encoders: dict):
    """Apply existing encoders to a dataframe (for inference)."""
    df2 = df.copy()
    for col, le in encoders.items():
        if col in df2.columns:
            # If unseen labels exist, map to -1 then to a new int
            try:
                df2[col] = le.transform(df2[col])
            except ValueError:
                # handle unseen labels by mapping unseen to a new integer
                mapping = {label: i for i, label in enumerate(le.classes_)}
                df2[col] = df2[col].map(lambda x: mapping.get(x, -1))
    return df2

def split_data(df: pd.DataFrame, target_column: str, test_size=0.2, random_state=42):
    """Split features and target; uses stratify if classification target is categorical."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    stratify = y if (y.dtype.name == "category" or y.dtype == object or (np.issubdtype(y.dtype, np.integer) and len(np.unique(y))<50)) else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standardize numeric features (returns numpy arrays and saved scaler)."""
    scaler = StandardScaler()
    # convert to numeric-only arrays (scaler requires numeric)
    X_train_arr = np.array(X_train, dtype=float)
    X_test_arr = np.array(X_test, dtype=float)
    X_train_scaled = scaler.fit_transform(X_train_arr)
    X_test_scaled = scaler.transform(X_test_arr)
    return X_train_scaled, X_test_scaled, scaler

def save_scaler(scaler, path="models/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def save_encoders(encoders: dict, path="models/encoders.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoders, path)

def load_scaler(path="models/scaler.pkl"):
    return joblib.load(path)

def load_encoders(path="models/encoders.pkl"):
    return joblib.load(path)
