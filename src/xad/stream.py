"""Row-by-row scoring for the streaming demo.

Kafka is kept out of this module so the logic can be tested without a broker;
`scripts/stream_producer.py` and `scripts/stream_consumer.py` are thin wrappers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np
import pandas as pd

from xad.baseline import fit_detector
from xad.compare import PROTOCOL, make_eif
from xad.data import SENSOR_COLUMNS

END_OF_STREAM = {"end": True}


def encode(message: dict) -> bytes:
    return json.dumps(message).encode("utf-8")


def decode(value: bytes) -> dict:
    message = json.loads(value)
    if not isinstance(message, dict):
        raise TypeError("message is not a JSON object")
    return message


def experiment_messages(df_exp: pd.DataFrame) -> Iterator[dict]:
    """One message per row: sensor values, plus the label (used only for reporting)."""
    for i, (timestamp, row) in enumerate(df_exp.iterrows()):
        yield {
            "row": i,
            "datetime": timestamp.isoformat(),
            "sensors": {col: float(row[col]) for col in SENSOR_COLUMNS},
            "anomaly": int(row["anomaly"]),
        }


class StreamingDetector:
    """Score sensor rows one at a time with the EIF and the comparison protocol.

    The first `warmup` rows are assumed to be normal operation: the scaler and
    EIF are fit on the first 75 % of them and the threshold is the 0.95
    quantile of the scores on the rest, exactly as in `xad.compare`.
    """

    def __init__(self, warmup: int = 400, seed: int = 0, n_top: int = 3):
        if int(warmup * PROTOCOL["calib_frac"]) < 1:  # need >= 1 calibration row
            raise ValueError(f"warmup too short: {warmup} rows")
        self.warmup, self.seed, self.n_top = warmup, seed, n_top
        self._rows: list[np.ndarray] = []
        self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def update(self, sensors: dict) -> dict | None:
        """Add one row; `None` during warm-up, else the score, alarm flag and top sensors.

        Raises `KeyError` for a missing sensor and `ValueError` for a non-numeric
        or non-finite value; the detector state is unchanged in both cases.
        """
        x = np.array([float(sensors[col]) for col in SENSOR_COLUMNS])
        if not np.isfinite(x).all():
            raise ValueError("non-finite sensor value")
        if not self.ready:
            self._rows.append(x)
            if len(self._rows) == self.warmup:
                self.scaler, self.model, self.threshold = fit_detector(
                    np.array(self._rows),
                    seed=self.seed,
                    make_model=make_eif,
                    **PROTOCOL,
                )
                self._rows = []
            return None

        xs = self.scaler.transform(x[None, :])
        score = float(self.model.score_samples(xs)[0])
        alarm = score > self.threshold
        top = []
        if alarm:
            importance = self.model.explain(xs)[0]
            total = importance.sum()
            share = importance / total if total > 0 else importance
            top = [
                (SENSOR_COLUMNS[k], float(share[k]))
                for k in np.argsort(share)[::-1][: self.n_top]
            ]
        return {"score": score, "threshold": self.threshold, "alarm": alarm, "top": top}
