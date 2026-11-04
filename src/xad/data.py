"""Loading helpers for the SKAB (Skoltech Anomaly Benchmark) dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SENSOR_COLUMNS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]
LABEL_COLUMNS = ["anomaly", "changepoint"]
SCENARIOS = ("valve1", "valve2", "other")


def read_experiment(path: Path | str) -> pd.DataFrame:
    """Read one SKAB experiment CSV (semicolon separated) indexed by datetime."""
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"], index_col="datetime")
    df = df.sort_index()
    for col in LABEL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def load_skab(root: Path | str, scenarios: tuple[str, ...] = SCENARIOS) -> pd.DataFrame:
    """Load all labelled SKAB experiments into one frame.

    `root` is the SKAB repository folder (it contains `data/`). Adds the columns
    `scenario` (valve1, valve2, other) and `experiment` (e.g. valve1/0).
    """
    root = Path(root)
    frames = []
    for scenario in scenarios:
        for csv in sorted((root / "data" / scenario).glob("*.csv")):
            df = read_experiment(csv)
            df["scenario"] = scenario
            df["experiment"] = f"{scenario}/{csv.stem}"
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No SKAB CSV files found under {root / 'data'}")
    return pd.concat(frames)
