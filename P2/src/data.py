from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def load_wdbc_from_zip(zip_path: str | Path = "data_a1.zip") -> tuple[np.ndarray, np.ndarray, list[str]]:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}")

    with ZipFile(zip_path) as archive:
        if "data.csv" not in archive.namelist():
            raise FileNotFoundError("data.csv is missing from data_a1.zip")
        with archive.open("data.csv") as handle:
            df = pd.read_csv(handle, index_col=0)

    if "diagnosis" not in df.columns:
        raise ValueError("Expected a diagnosis column in data.csv")

    labels = df["diagnosis"].map({"M": 1.0, "B": -1.0})
    if labels.isna().any():
        raise ValueError("Diagnosis column contains labels other than M and B")
    y = labels.to_numpy(dtype=float)

    drop_cols = [col for col in ["id", "diagnosis"] if col in df.columns]
    feature_df = df.drop(columns=drop_cols)
    X = feature_df.to_numpy(dtype=float)
    return X, y, list(feature_df.columns)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(X.shape[0])
    n_test = int(round(X.shape[0] * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    standardizer = Standardizer(mean=mean, scale=scale)
    return standardizer.transform(X_train), standardizer.transform(X_test), mean, scale
