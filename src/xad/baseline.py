"""Isolation Forest baseline on SKAB.

Each experiment is handled separately: the model is fit on the leading normal
part of that experiment and evaluated on the rows after it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xad.data import SENSOR_COLUMNS


def make_features(df: pd.DataFrame, rolling: int = 0) -> pd.DataFrame:
    """Raw sensor channels, optionally with rolling mean and std features."""
    X = df[SENSOR_COLUMNS].astype(float)
    if rolling > 1:
        mean = X.rolling(rolling, min_periods=1).mean().add_suffix("_mean")
        std = X.rolling(rolling, min_periods=2).std().fillna(0.0).add_suffix("_std")
        X = pd.concat([X, mean, std], axis=1)
    return X


def train_end_index(labels, n_train: int = 400, margin: int = 5) -> int:
    """End of the known-normal training window (exclusive)."""
    labels = np.asarray(labels)
    anomalous = np.flatnonzero(labels == 1)
    first = int(anomalous[0]) if anomalous.size else len(labels)
    return max(10, min(n_train, first - margin))


def run_experiment(
    df_exp: pd.DataFrame,
    n_train: int = 400,
    rolling: int = 0,
    quantile: float = 0.99,
    seed: int = 0,
) -> dict:
    """Fit on the normal prefix, score the whole experiment, evaluate after it."""
    X = make_features(df_exp, rolling)
    y = df_exp["anomaly"].to_numpy()
    end = train_end_index(y, n_train)

    scaler = StandardScaler().fit(X.iloc[:end])
    Xs = scaler.transform(X)
    model = IsolationForest(n_estimators=200, random_state=seed).fit(Xs[:end])
    score = -model.score_samples(Xs)  # higher = more anomalous
    threshold = np.quantile(score[:end], quantile)
    pred = (score > threshold).astype(int)

    y_eval, pred_eval, score_eval = y[end:], pred[end:], score[end:]
    both_classes = len(np.unique(y_eval)) == 2
    return {
        "n_train": end,
        "precision": precision_score(y_eval, pred_eval, zero_division=0),
        "recall": recall_score(y_eval, pred_eval, zero_division=0),
        "f1": f1_score(y_eval, pred_eval, zero_division=0),
        "roc_auc": roc_auc_score(y_eval, score_eval) if both_classes else np.nan,
    }


def evaluate_all(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run every experiment in a SKAB frame and return one row per experiment."""
    rows = [
        {"experiment": name, **run_experiment(group, **kwargs)}
        for name, group in df.groupby("experiment", sort=True)
    ]
    return pd.DataFrame(rows).set_index("experiment")
