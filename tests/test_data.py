import pandas as pd
import pytest

from xad.data import SENSOR_COLUMNS, load_skab, read_experiment

HEADER = "datetime;" + ";".join(SENSOR_COLUMNS) + ";anomaly;changepoint"
ROW = "2020-03-09 10:14:33;0.03;0.04;1.0;0.5;70.0;24.0;220.0;32.0;{a}.0;0.0"


def _write(path, anomalies):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [ROW.format(a=a) for a in anomalies]
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n")


def test_read_experiment(tmp_path):
    csv = tmp_path / "0.csv"
    _write(csv, [0, 1])
    df = read_experiment(csv)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df["anomaly"]) == [0, 1]
    assert set(SENSOR_COLUMNS) <= set(df.columns)


def test_load_skab(tmp_path):
    _write(tmp_path / "data" / "valve1" / "0.csv", [0, 1])
    _write(tmp_path / "data" / "other" / "5.csv", [0])
    df = load_skab(tmp_path)
    assert len(df) == 3
    assert set(df["experiment"]) == {"valve1/0", "other/5"}


def test_load_skab_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_skab(tmp_path)
