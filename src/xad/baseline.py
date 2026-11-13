"""Isolation Forest baseline on SKAB.

Each experiment is handled separately: the model is fit on the leading normal
part of that experiment and evaluated on the rows after it.
"""

from __future__ import annotations

from dataclasses import dataclass

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


class IsolationForestDetector:
    """scikit-learn's Isolation Forest with `score_samples` flipped: higher = more anomalous."""

    def __init__(self, random_state: int | None = None):
        self.model = IsolationForest(n_estimators=200, random_state=random_state)

    def fit(self, X) -> IsolationForestDetector:
        self.model.fit(X)
        return self

    def score_samples(self, X) -> np.ndarray:
        return -self.model.score_samples(X)


@dataclass
class ExperimentRun:
    """A detector fitted on one experiment's normal prefix, with scores for every row."""

    model: object
    X: np.ndarray  # standardised features of all rows
    y: np.ndarray  # anomaly labels
    end: int  # first evaluated row (end of the normal prefix)
    score: np.ndarray  # higher = more anomalous
    threshold: float


def fit_detector(
    X_prefix,
    quantile: float = 0.99,
    seed: int = 0,
    calib_frac: float = 0.0,
    make_model=IsolationForestDetector,
) -> tuple[StandardScaler, object, float]:
    """Fit a scaler, a detector and an alarm threshold on known-normal rows.

    With `calib_frac > 0` the last `calib_frac` of the rows is held out: the
    scaler and model are fit on the rest, and the threshold is the `quantile`
    of the held-out scores (in-sample scores are optimistically low).
    `make_model(random_state)` returns an unfitted detector with `fit` and a
    `score_samples` where higher = more anomalous.
    """
    if not 0.0 <= calib_frac < 1.0:
        raise ValueError(f"calib_frac must be in [0, 1), got {calib_frac}")
    X_prefix = np.asarray(X_prefix, dtype=float)
    fit_end = len(X_prefix) - int(len(X_prefix) * calib_frac)
    scaler = StandardScaler().fit(X_prefix[:fit_end])
    Xs = scaler.transform(X_prefix)
    model = make_model(seed).fit(Xs[:fit_end])
    calib = Xs[fit_end:] if fit_end < len(Xs) else Xs
    return scaler, model, float(np.quantile(model.score_samples(calib), quantile))


def fit_experiment(
    df_exp: pd.DataFrame,
    n_train: int = 400,
    rolling: int = 0,
    **fit_kwargs,
) -> ExperimentRun:
    """Fit with `fit_detector(**fit_kwargs)` on the normal prefix; score the whole experiment."""
    X = make_features(df_exp, rolling).to_numpy(dtype=float)
    y = df_exp["anomaly"].to_numpy()
    end = train_end_index(y, n_train)
    scaler, model, threshold = fit_detector(X[:end], **fit_kwargs)
    Xs = scaler.transform(X)
    return ExperimentRun(model, Xs, y, end, model.score_samples(Xs), threshold)


def run_experiment(df_exp: pd.DataFrame, **kwargs) -> dict:
    """Fit with `fit_experiment(**kwargs)` and compute metrics on the rows after the prefix."""
    run = fit_experiment(df_exp, **kwargs)
    pred = (run.score > run.threshold).astype(int)
    y_eval, pred_eval, score_eval = (
        run.y[run.end :],
        pred[run.end :],
        run.score[run.end :],
    )
    both_classes = len(np.unique(y_eval)) == 2
    return {
        "n_train": run.end,
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
