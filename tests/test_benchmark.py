import numpy as np
import pytest
from test_baseline import _synthetic

from xad.benchmark import (
    FIT_END,
    N_TRAIN,
    RATE_LAG,
    evaluate,
    false_alarm_positions,
    features,
    pooled_metrics,
    predict,
    raw_scores,
    split_experiments,
)


def test_pooled_metrics_matches_skab_formulas():
    # experiment 1: TP=2, FN=1, FP=1, TN=1; experiment 2: TP=1, FP=0, FN=0, TN=2
    pairs = [([1, 1, 1, 0, 0], [1, 1, 0, 1, 0]), ([1, 0, 0], [1, 0, 0])]
    m = pooled_metrics(pairs)
    tp, fp, fn, tn = 3, 1, 1, 3
    assert m["f1"] == pytest.approx(tp / (tp + (fn + fp) / 2))
    assert m["far"] == pytest.approx(100 * fp / (fp + tn))
    assert m["mar"] == pytest.approx(100 * fn / (fn + tp))


def test_detrend_removes_linear_drift_fitted_on_training_rows():
    df = _synthetic()
    t = np.arange(len(df))
    df["Thermocouple"] += 0.01 * t
    X = features(df, "detrend")
    col = list(df.columns).index("Thermocouple")
    # drift removed: the late normal rows are not shifted relative to the early ones
    assert abs(X[650:, col].mean() - X[:FIT_END, col].mean()) < 0.3
    with pytest.raises(ValueError):
        features(df, "unknown")


def test_thermocouple_variants():
    df = _synthetic()
    col = list(df.columns).index("Thermocouple")
    df["Thermocouple"] = 25.0 + 0.1 * np.arange(len(df))  # steady heating
    raw = features(df, "raw")
    assert features(df, "no_thermo").shape == (len(df), raw.shape[1] - 1)
    rate = features(df, "thermo_rate")
    np.testing.assert_allclose(rate[RATE_LAG:, col], 0.1 * RATE_LAG)  # level removed
    assert rate[0, col] == 0.0
    np.testing.assert_array_equal(np.delete(rate, col, 1), np.delete(raw, col, 1))


def test_false_alarm_positions():
    y = np.zeros(N_TRAIN + 10, dtype=int)
    y[N_TRAIN + 3 : N_TRAIN + 6] = 1  # anomaly rows 403-405
    pred = np.zeros_like(y, dtype=bool)
    pred[[0, N_TRAIN + 1, N_TRAIN + 4, N_TRAIN + 7, N_TRAIN + 8]] = True
    out = false_alarm_positions(y, pred)
    assert (
        out["normal_before"] == 3 and out["false_alarms_before"] == 1
    )  # row 0 is training
    assert out["normal_after"] == 4 and out["false_alarms_after"] == 2
    assert out["normal_between"] == 0


def test_predict_is_causal_and_uses_calibration_rows():
    rng = np.random.default_rng(0)
    score = rng.normal(size=600)
    changed = score.copy()
    changed[500:] += 10
    a, b = predict(score, 15, 0.99), predict(changed, 15, 0.99)
    np.testing.assert_array_equal(a[:500], b[:500])
    assert b[500:].all()
    # the threshold is the calibration quantile, so at most 1 % of them alarm
    assert a[FIT_END:N_TRAIN].mean() <= 0.01


def test_evaluate_on_synthetic_experiments():
    dfs = {"valve1/0": _synthetic(seed=0), "other/0": _synthetic(seed=1)}
    labels = {e: d["anomaly"].to_numpy() for e, d in dfs.items()}
    scores = {e: raw_scores(d, "EIF", "raw") for e, d in dfs.items()}
    m = evaluate(scores, labels, list(dfs), smooth=1, quantile=0.99)
    assert m["f1"] > 0.8
    assert split_experiments(dfs) == (["valve1/0"], ["other/0"])
