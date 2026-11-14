"""Synthetic ground-truth test of per-sensor explanations on real SKAB normal rows.

Per experiment and seed: scaler, EIF and Isolation Forest are fit on the first
300 rows (as in the SKAB-protocol benchmark); anomalies with known responsible
sensors are injected into the next 100 normal rows (see xad.attribution).

Writes docs/explanation_benchmark.csv (per experiment, seed, task, method) and
docs/explanation_benchmark_summary.csv (mean over experiments, then seeds).
"""

import numpy as np
import pandas as pd

from xad.attribution import (
    correlated_pair,
    inject_shift,
    inject_swap,
    local_diffi,
    precision_at_k,
    zscore_importance,
)
from xad.baseline import IsolationForestDetector, fit_detector
from xad.benchmark import FIT_END, N_TRAIN
from xad.compare import PROTOCOL, make_eif
from xad.data import SENSOR_COLUMNS, load_skab

SEEDS = range(3)
SHIFT_K = (1, 2, 3)
SHIFT_MAGNITUDE = (2.0, 3.0, 5.0)
MIN_CORR = (
    0.6  # swap test only where the strongest sensor pair is at least this correlated
)

df = load_skab("data/raw/SKAB")
records = []
for name, g in df.groupby("experiment", sort=True):
    X = g[SENSOR_COLUMNS].to_numpy(dtype=float)[:N_TRAIN]
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        scaler, eif, eif_thr = fit_detector(
            X, seed=seed, make_model=make_eif, **PROTOCOL
        )
        _, iforest, if_thr = fit_detector(
            X, seed=seed, make_model=IsolationForestDetector, **PROTOCOL
        )
        Xs = scaler.transform(X)
        X_fit, base = Xs[:FIT_END], Xs[FIT_END:]

        tasks = [
            (f"shift k={k} m={m:g}", *inject_shift(base, k, m, rng))
            for k in SHIFT_K
            for m in SHIFT_MAGNITUDE
        ]
        a, b, corr = correlated_pair(X_fit)
        if abs(corr) >= MIN_CORR:
            rows, truth = inject_swap(base, X_fit, a, b, rng)
            if len(rows):
                tasks.append(("swap", rows, truth))

        for task, rows, truth in tasks:
            detected = {
                "EIF": (eif.score_samples(rows) > eif_thr).mean(),
                "IF": (iforest.score_samples(rows) > if_thr).mean(),
            }
            methods = {
                ("EIF explain (ours)", "EIF"): eif.explain(rows),
                ("Local-DIFFI", "IF"): local_diffi(iforest.model, rows),
                ("z-score", "-"): zscore_importance(rows),
                ("random", "-"): rng.random(rows.shape),
            }
            for (method, model), importance in methods.items():
                records.append(
                    {
                        "experiment": name,
                        "seed": seed,
                        "task": task,
                        "method": method,
                        "n_rows": len(rows),
                        "precision": precision_at_k(importance, truth).mean(),
                        "model_detection_rate": detected.get(model, np.nan),
                    }
                )

results = pd.DataFrame(records)
results.to_csv("docs/explanation_benchmark.csv", index=False, float_format="%.4f")

per_seed = results.groupby(["task", "method", "seed"], sort=False)[
    ["precision", "model_detection_rate"]
].mean()  # experiments weighted equally
summary = per_seed.groupby(["task", "method"], sort=False).agg(
    precision=("precision", "mean"),
    precision_min=("precision", "min"),
    precision_max=("precision", "max"),
    detection_rate=("model_detection_rate", "mean"),
)
n_swap = results.loc[results.task == "swap", "experiment"].nunique()
summary.to_csv("docs/explanation_benchmark_summary.csv", float_format="%.4f")

print(
    f"experiments: {results.experiment.nunique()}, swap test on {n_swap} (|corr| >= {MIN_CORR})"
)
print(
    "precision@k = share of the truly responsible sensors among the top-k, k = their number"
)
print(summary["precision"].unstack("method").round(3).to_string())
print("\nmodel detection rate of the injected rows:")
print(
    summary["detection_rate"]
    .unstack("method")[["EIF explain (ours)", "Local-DIFFI"]]
    .round(3)
    .to_string()
)
