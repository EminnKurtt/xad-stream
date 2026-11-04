import numpy as np
import pandas as pd

from xad.baseline import make_features, run_experiment, train_end_index
from xad.data import SENSOR_COLUMNS


def _synthetic(n=700, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(n, len(SENSOR_COLUMNS))),
        columns=SENSOR_COLUMNS,
        index=pd.date_range("2020-01-01", periods=n, freq="s"),
    )
    df["anomaly"] = 0
    df["changepoint"] = 0
    for col in ("Temperature", "Thermocouple", "Current"):
        df.iloc[450:600, df.columns.get_loc(col)] += 4.0
    df.iloc[450:600, df.columns.get_loc("anomaly")] = 1
    return df


def test_train_end_index():
    labels = np.zeros(1000, dtype=int)
    labels[104:] = 1
    assert train_end_index(labels) == 99
    labels = np.zeros(1000, dtype=int)
    labels[570:] = 1
    assert train_end_index(labels) == 400
    assert train_end_index(np.zeros(200, dtype=int)) == 195


def test_make_features_shapes():
    df = _synthetic()
    assert make_features(df).shape == (700, 8)
    assert make_features(df, rolling=30).shape == (700, 24)


def test_run_experiment_detects_multichannel_shift():
    result = run_experiment(_synthetic())
    assert result["n_train"] == 400
    assert result["roc_auc"] > 0.95
    assert result["f1"] > 0.8
