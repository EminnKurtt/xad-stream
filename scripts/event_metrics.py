"""Operator-level view of the SKAB-protocol detectors: events, delays, false alarms per hour.

Default post-processing (no smoothing, 0.95 quantile). Alarms at most MERGE_GAP
quiet seconds apart form one event. Writes docs/event_metrics.csv (per config,
seed, experiment) and docs/event_metrics_summary.csv (mean over seeds).
"""

import numpy as np
import pandas as pd

from xad.benchmark import N_TRAIN, event_metrics, predict, raw_scores
from xad.data import load_skab

SEEDS = range(3)
MERGE_GAP = 10
CONFIGS = {
    "EIF, raw sensors": ("EIF", "raw"),
    "EIF, without Thermocouple": ("EIF", "no_thermo"),
    "IF, raw sensors": ("IF", "raw"),
}

df = load_skab("data/raw/SKAB")
groups = dict(tuple(df.groupby("experiment", sort=True)))
rows = []
SHUFFLED = "EIF, raw, alarms shuffled in time"
for config, (model, kind) in CONFIGS.items():
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for name, g in groups.items():
            y = g["anomaly"].to_numpy()
            pred = predict(raw_scores(g, model, kind, seed), smooth=1, quantile=0.95)
            m = event_metrics(y, pred, MERGE_GAP)
            rows.append({"config": config, "seed": seed, "experiment": name, **m})
            if config == "EIF, raw sensors":
                # reference: same number of test alarms, at random times
                shuffled = pred.copy()
                shuffled[N_TRAIN:] = rng.permutation(pred[N_TRAIN:])
                m = event_metrics(y, shuffled, MERGE_GAP)
                rows.append({"config": SHUFFLED, "seed": seed, "experiment": name, **m})
for name, g in groups.items():  # reference: an alarm on every row
    y = g["anomaly"].to_numpy()
    m = event_metrics(y, np.ones(len(y), dtype=bool), MERGE_GAP)
    rows.append({"config": "alarm on every row", "seed": 0, "experiment": name, **m})

per_exp = pd.DataFrame(rows)
per_exp.assign(delays=per_exp["delays"].map(lambda d: "|".join(map(str, d)))).to_csv(
    "docs/event_metrics.csv", index=False
)


def summarise(part: pd.DataFrame) -> pd.Series:
    delays = [d for ds in part["delays"] for d in ds]
    hours_before = part["normal_s_before"].sum() / 3600
    hours_after = part["normal_s_after"].sum() / 3600
    return pd.Series(
        {
            "detected_pct": 100 * part["detected"].sum() / part["windows"].sum(),
            "median_delay_s": np.median(delays) if delays else np.nan,
            "false_events_per_h_before": part["false_events_before"].sum()
            / hours_before,
            "false_events_per_h_after": part["false_events_after"].sum() / hours_after,
            "false_events_per_h": (
                part["false_events_before"].sum() + part["false_events_after"].sum()
            )
            / (hours_before + hours_after),
        }
    )


per_seed = per_exp.groupby(["config", "seed"], sort=False).apply(
    summarise, include_groups=False
)
summary = per_seed.groupby("config", sort=False).mean()
summary.to_csv("docs/event_metrics_summary.csv", float_format="%.2f")

n_windows = per_exp.loc[per_exp.seed == 0].groupby("config")["windows"].sum().iloc[0]
hours = (
    per_exp.loc[
        per_exp.config == "alarm on every row", ["normal_s_before", "normal_s_after"]
    ].sum()
    / 3600
)
print(
    f"{n_windows} anomaly windows; normal test time {hours.sum():.2f} h "
    f"({hours['normal_s_before']:.2f} h before, {hours['normal_s_after']:.2f} h after the anomaly)"
)
print(f"events: alarms at most {MERGE_GAP} s apart; mean over seeds {list(SEEDS)}")
print(summary.round(2).to_string())
