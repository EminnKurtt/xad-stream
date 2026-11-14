"""Evaluation with the SKAB benchmark protocol, comparable to its leaderboard.

Protocol, as in the SKAB repository (`core/utils.py`, `core/metrics.py`): in
every experiment the first 400 rows are the training set and the remaining
rows the test set; labels are never used for training. Metrics pool all test
rows of all experiments:
`F1 = TP / (TP + (FN + FP) / 2)`, `FAR = FP / (FP + TN)`, `MAR = FN / (FN + TP)`.

Inside the 400 training rows our detectors fit on the first 300 and set the
alarm threshold on the last 100 (the calibration rows), so every choice below
is made from training data only:

- `features`: raw sensors, or sensors minus a per-sensor linear trend fitted on
  the 300 fitting rows (targets slow drift, e.g. the fluid heating up).
- `smooth`: causal moving average of the score over the last `smooth` rows.
- `quantile`: of the smoothed calibration scores, used as alarm threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xad.baseline import IsolationForestDetector, fit_detector
from xad.compare import PROTOCOL, make_eif
from xad.data import SENSOR_COLUMNS

N_TRAIN = 400
FIT_END = N_TRAIN - int(N_TRAIN * PROTOCOL["calib_frac"])  # 300
MODELS = {"IF": IsolationForestDetector, "EIF": make_eif}
FEATURES = ("raw", "detrend")
SMOOTH = (1, 5, 15, 31)
QUANTILES = (0.95, 0.99, 1.0)
DEV_SCENARIOS = ("valve1",)  # choices are made here; valve2 + other are held out


def features(df_exp: pd.DataFrame, kind: str) -> np.ndarray:
    X = df_exp[SENSOR_COLUMNS].to_numpy(dtype=float)
    if kind == "detrend":
        t = np.arange(len(X), dtype=float)
        slope, intercept = np.polyfit(t[:FIT_END], X[:FIT_END], 1)
        return X - (np.outer(t, slope) + intercept)
    if kind != "raw":
        raise ValueError(f"unknown features: {kind}")
    return X


def raw_scores(
    df_exp: pd.DataFrame, model: str, kind: str, seed: int = 0
) -> np.ndarray:
    """Anomaly scores for every row; scaler and model see only the first 300 rows."""
    X = features(df_exp, kind)
    scaler, detector, _ = fit_detector(
        X[:N_TRAIN], seed=seed, make_model=MODELS[model], **PROTOCOL
    )
    return detector.score_samples(scaler.transform(X))


def predict(score: np.ndarray, smooth: int, quantile: float) -> np.ndarray:
    """Alarms for every row: causal smoothing, threshold from the calibration rows."""
    smoothed = pd.Series(score).rolling(smooth, min_periods=1).mean().to_numpy()
    return smoothed > np.quantile(smoothed[FIT_END:N_TRAIN], quantile)


def pooled_metrics(pairs) -> dict:
    """SKAB binary metrics over `(y_true, y_pred)` pairs of test rows (FAR, MAR in %)."""
    tp = fp = fn = tn = 0
    for y_true, y_pred in pairs:
        t, p = np.asarray(y_true) == 1, np.asarray(y_pred, dtype=bool)
        tp += int((t & p).sum())
        fp += int((~t & p).sum())
        fn += int((t & ~p).sum())
        tn += int((~t & ~p).sum())
    return {
        "f1": tp / (tp + (fn + fp) / 2),
        "far": 100 * fp / (fp + tn),
        "mar": 100 * fn / (fn + tp),
    }


def evaluate(
    scores: dict, labels: dict, experiments, smooth: int, quantile: float
) -> dict:
    """Pooled test metrics of one post-processing setting over `experiments`."""
    return pooled_metrics(
        (labels[e][N_TRAIN:], predict(scores[e], smooth, quantile)[N_TRAIN:])
        for e in experiments
    )


def split_experiments(names) -> tuple[list[str], list[str]]:
    dev = sorted(n for n in names if n.split("/")[0] in DEV_SCENARIOS)
    test = sorted(n for n in names if n.split("/")[0] not in DEV_SCENARIOS)
    return dev, test
