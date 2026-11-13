import numpy as np
import pytest
from test_baseline import _synthetic

from xad.baseline import fit_experiment
from xad.compare import PROTOCOL, make_eif
from xad.data import SENSOR_COLUMNS
from xad.stream import (
    END_OF_STREAM,
    StreamingDetector,
    decode,
    encode,
    experiment_messages,
)

SHIFTED = {"Temperature", "Thermocouple", "Current"}


def test_messages_round_trip():
    df = _synthetic(n=5)
    messages = list(experiment_messages(df))
    assert [m["row"] for m in messages] == [0, 1, 2, 3, 4]
    assert set(messages[0]["sensors"]) == set(SENSOR_COLUMNS)
    assert decode(encode(messages[0])) == messages[0]
    assert decode(encode(END_OF_STREAM)) == END_OF_STREAM
    with pytest.raises(TypeError):
        decode(b"[1, 2]")


def test_streaming_matches_batch_protocol():
    # anomaly starts at row 450, so the batch prefix is exactly the 400-row warm-up
    df = _synthetic()
    detector = StreamingDetector(warmup=400)
    results = [detector.update(m["sensors"]) for m in experiment_messages(df)]
    assert all(r is None for r in results[:400])
    assert detector.ready

    batch = fit_experiment(df, make_model=make_eif, **PROTOCOL)
    np.testing.assert_allclose([r["score"] for r in results[400:]], batch.score[400:])
    assert detector.threshold == pytest.approx(batch.threshold)


def test_alarms_name_the_shifted_sensors():
    df = _synthetic()
    detector = StreamingDetector()
    results = [detector.update(m["sensors"]) for m in experiment_messages(df)]
    in_window = [r for r in results[450:600] if r["alarm"]]
    assert len(in_window) > 0.9 * 150
    assert all(len(r["top"]) == 3 for r in in_window)
    top1 = [r["top"][0][0] for r in in_window]
    assert np.mean([s in SHIFTED for s in top1]) > 0.9


def test_bad_rows_are_rejected_without_changing_state():
    detector = StreamingDetector(warmup=4)
    row = {col: 1.0 for col in SENSOR_COLUMNS}
    with pytest.raises(KeyError):
        detector.update({"Temperature": 1.0})
    with pytest.raises(ValueError):
        detector.update({**row, "Current": float("nan")})
    with pytest.raises(ValueError):
        detector.update({**row, "Current": "abc"})
    assert detector._rows == []
    with pytest.raises(ValueError):
        StreamingDetector(warmup=3)
