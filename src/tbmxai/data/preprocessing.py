"""Leakage-safe tabular preprocessing.

Preprocessors are *unfitted* sklearn transformers; models fit them on the
training split only (see :class:`tbmxai.models.tabular.SklearnTabularClassifier`).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into (numeric, categorical) by dtype.

    Booleans count as numeric; object/category/string columns as categorical.
    """
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    return numeric, categorical


def build_tabular_preprocessor(
    frame: pd.DataFrame, scale_numeric: bool = True
) -> ColumnTransformer:
    """Create an (unfitted) imputer + scaler + one-hot encoder for ``frame``.

    Args:
        frame: Training features; dtypes decide numeric vs categorical, and columns
            with no observed value are dropped.
        scale_numeric: Standardise numeric columns (disable for tree models).

    Returns:
        A ``ColumnTransformer`` producing a dense float matrix.
    """
    observed = frame.columns[frame.notna().any()]
    numeric, categorical = split_feature_types(frame[observed])  # all-missing columns are dropped
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
