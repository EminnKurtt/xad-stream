from test_baseline import _synthetic

from xad.explain import anomaly_windows, explain_experiment, sensor_summary

SHIFTED = {"Temperature", "Thermocouple", "Current"}


def test_anomaly_windows():
    assert anomaly_windows([0, 1, 1, 0, 0, 1]) == [(1, 3), (5, 6)]
    assert anomaly_windows([1, 1]) == [(0, 2)]
    assert anomaly_windows([0, 0]) == []


def test_sensor_summary_points_to_shifted_sensors():
    df = _synthetic()
    run, importance = explain_experiment(df)
    assert importance.shape == (len(df), 8)
    summary = sensor_summary(df, run, importance)
    assert set(summary["importance_in"].nlargest(3).index) == SHIFTED
    assert set(summary["shift"].nlargest(3).index) == SHIFTED
    shifted = summary.loc[sorted(SHIFTED)]
    assert (shifted["importance_in"] > shifted["importance_out"]).all()
