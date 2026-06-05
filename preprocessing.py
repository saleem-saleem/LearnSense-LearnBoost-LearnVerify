"""
preprocessing.py  -  Data loading and preprocessing utilities
=============================================================
"""

from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


DATASET_URLS = {
    "D1": "https://www.kaggle.com/datasets/mitgandhi10/dataset-for-multiple-regression",
    "D2": "https://www.kaggle.com/datasets/lainguyn123/student-performance-factors",
    "D3": "https://archive.ics.uci.edu/dataset/320/student+performance",
    "D4": "https://archive.ics.uci.edu/dataset/582/student+performance+on+an+entrance+examination",
    "D5": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
}

SEEDS = [42, 77, 101, 202, 555]


def encode_and_scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Label-encode categorical columns and MinMax-scale numeric columns.
    Fit ONLY on X_train; apply to both splits (leakage-free).
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    encoders, scalers = {}, {}

    for col in X_tr.columns:
        if X_tr[col].dtype == object or str(X_tr[col].dtype) == "category":
            le = LabelEncoder()
            X_tr[col] = le.fit_transform(X_tr[col].astype(str))
            X_te[col] = le.transform(X_te[col].astype(str))
            encoders[col] = le
        else:
            sc = MinMaxScaler()
            X_tr[[col]] = sc.fit_transform(X_tr[[col]])
            X_te[[col]] = sc.transform(X_te[[col]])
            scalers[col] = sc

    return X_tr.values, X_te.values, encoders, scalers


def make_splits(X, y, seeds=SEEDS, test_size=0.20):
    """
    Create seed-indexed 80:20 stratified train-test splits.

    Returns
    -------
    list of (X_train, X_test, y_train, y_test) tuples, one per seed.
    """
    splits = []
    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=seed)
        tr_idx, te_idx = next(sss.split(X, y))
        X_tr_raw, X_te_raw = (X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx],
                               X.iloc[te_idx] if hasattr(X, "iloc") else X[te_idx])
        y_tr = (y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx])
        y_te = (y.iloc[te_idx] if hasattr(y, "iloc") else y[te_idx])

        if hasattr(X_tr_raw, "columns"):
            X_tr, X_te, _, _ = encode_and_scale(
                pd.DataFrame(X_tr_raw), pd.DataFrame(X_te_raw))
        else:
            X_tr, X_te = X_tr_raw.astype(float), X_te_raw.astype(float)

        splits.append((X_tr, X_te,
                        np.asarray(y_tr), np.asarray(y_te)))
    return splits


def save_splits(splits, dataset_name: str, out_dir: str = "data/splits"):
    """Save train/test index arrays as JSON for reproducibility."""
    os.makedirs(out_dir, exist_ok=True)
    data = {}
    for seed, (Xtr, Xte, _, _) in zip(SEEDS, splits):
        data[str(seed)] = {
            "n_train": len(Xtr),
            "n_test":  len(Xte),
        }
    path = os.path.join(out_dir, f"{dataset_name}_splits.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Split metadata saved to {path}")
