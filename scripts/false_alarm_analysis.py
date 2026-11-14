"""Where do the false alarms fall, and do the sensors return to their training level?

SKAB protocol, default post-processing (no smoothing, 0.95 quantile), seed 0.
Writes docs/false_alarm_positions.csv and docs/post_anomaly_shift.csv.
"""

import sys

import numpy as np
import pandas as pd

from xad.benchmark import N_TRAIN, false_alarm_positions, predict, raw_scores
from xad.data import SENSOR_COLUMNS, load_skab

FEATURE_SETS = sys.argv[1:] or ["raw"]
df = load_skab("data/raw/SKAB")
groups = dict(tuple(df.groupby("experiment", sort=True)))

rows = []
for kind in FEATURE_SETS:
    for name, g in groups.items():
        pred = predict(raw_scores(g, "EIF", kind), smooth=1, quantile=0.95)
        y = g["anomaly"].to_numpy()
        rows.append(
            {"features": kind, "experiment": name, **false_alarm_positions(y, pred)}
        )
positions = pd.DataFrame(rows)
positions.to_csv("docs/false_alarm_positions.csv", index=False)

# shift of each sensor after the last anomaly row, in units of its training-row std
shift = []
for name, g in groups.items():
    X = g[SENSOR_COLUMNS].to_numpy(dtype=float)
    last = np.flatnonzero(g["anomaly"].to_numpy() == 1)[-1]
    if len(X) - last - 1 >= 30:
        train = X[:N_TRAIN]
        after = X[last + 1 :]
        shift.append(np.abs(after.mean(0) - train.mean(0)) / train.std(0))
n_shift = len(shift)
shift = pd.Series(
    np.mean(shift, axis=0), index=SENSOR_COLUMNS, name="mean_shift_in_train_std"
)
shift.to_frame().to_csv("docs/post_anomaly_shift.csv", float_format="%.2f")

print("== EIF false alarms by position (normal test rows), summed over experiments ==")
for kind, part in positions.groupby("features", sort=False):
    total = part.drop(columns=["features", "experiment"]).sum()
    all_fa = sum(total[f"false_alarms_{p}"] for p in ("before", "between", "after"))
    print(f"[{kind}]")
    for p in ("before", "between", "after"):
        n, fa = total[f"normal_{p}"], total[f"false_alarms_{p}"]
        if n:
            print(
                f"  {p:7s}: {fa:5d} of {n:5d} normal rows flagged ({100 * fa / n:.0f} %), "
                f"{100 * fa / all_fa:.0f} % of all false alarms"
            )
print(
    f"\n== mean shift after the anomaly, in training-row std ({n_shift} experiments) =="
)
print(shift.round(1).to_string())
