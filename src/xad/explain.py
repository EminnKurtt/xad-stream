"""Sensor-level explanations of the EIF detector on SKAB experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xad.baseline import ExperimentRun, fit_experiment
from xad.compare import PROTOCOL, make_eif
from xad.data import SENSOR_COLUMNS


def anomaly_windows(labels) -> list[tuple[int, int]]:
    """Contiguous runs of `labels == 1` as `(start, stop)` row ranges (stop exclusive)."""
    edges = np.diff(np.r_[0, np.asarray(labels, dtype=int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def explain_experiment(
    df_exp: pd.DataFrame, seed: int = 0
) -> tuple[ExperimentRun, np.ndarray]:
    """Fit the EIF with the comparison protocol; return the run and `explain` on every row."""
    run = fit_experiment(df_exp, seed=seed, make_model=make_eif, **PROTOCOL)
    return run, run.model.explain(run.X)


def sensor_summary(
    df_exp: pd.DataFrame, run: ExperimentRun, importance: np.ndarray
) -> pd.DataFrame:
    """Per-sensor importance inside / outside the anomaly windows, on the evaluated rows.

    `importance_*` is the raw `explain` output. It shrinks for easy-to-isolate
    rows, because their paths are shorter and collect fewer splits, so
    `share_*` also normalises each row to sum to 1 before averaging.
    `shift` is a model-free reference for how visibly a sensor changes:
    `|mean inside - mean outside| / std` of the raw values over the evaluated rows.
    """
    y = run.y[run.end :]
    imp = importance[run.end :]
    total = imp.sum(axis=1, keepdims=True)
    share = np.divide(imp, total, out=np.zeros_like(imp), where=total > 0)
    raw = df_exp[SENSOR_COLUMNS].to_numpy(dtype=float)[run.end :]
    inside, outside = y == 1, y == 0
    std = raw.std(axis=0)
    shift = np.abs(raw[inside].mean(axis=0) - raw[outside].mean(axis=0))
    return pd.DataFrame(
        {
            "importance_in": imp[inside].mean(axis=0),
            "importance_out": imp[outside].mean(axis=0),
            "share_in": share[inside].mean(axis=0),
            "share_out": share[outside].mean(axis=0),
            "shift": np.divide(shift, std, out=np.zeros_like(shift), where=std > 0),
        },
        index=pd.Index(SENSOR_COLUMNS, name="sensor"),
    )
