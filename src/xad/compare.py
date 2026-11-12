"""Compare Isolation Forest and Extended Isolation Forest on SKAB.

Both models use the same per-experiment protocol: fit on the first 75 % of the
known-normal prefix, set the threshold to the 0.95 quantile of the scores on
the held-out last 25 %, and compute metrics on the rows after the prefix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xad.baseline import isolation_forest_scorer, run_experiment, train_end_index
from xad.eif import ExtendedIsolationForest

PROTOCOL = {"calib_frac": 0.25, "quantile": 0.95}
METRICS = ["precision", "recall", "f1", "roc_auc"]


def eif_scorer(X_fit, seed: int = 0):
    """Fit a fully extended EIF with the same tree count as the Isolation Forest."""
    return (
        ExtendedIsolationForest(n_trees=200, random_state=seed).fit(X_fit).score_samples
    )


MODELS = {"IF": isolation_forest_scorer, "EIF": eif_scorer}


def compare_models(df: pd.DataFrame, models=MODELS, seed: int = 0) -> pd.DataFrame:
    """One row per (model, experiment) with the evaluation metrics."""
    rows = [
        {
            "model": model,
            "experiment": name,
            **run_experiment(group, seed=seed, fit_scorer=fit_scorer, **PROTOCOL),
        }
        for model, fit_scorer in models.items()
        for name, group in df.groupby("experiment", sort=True)
    ]
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Mean metrics per model and the number of experiments where EIF beats IF on F1."""
    means = results.groupby("model", sort=False)[METRICS].mean()
    f1 = results.pivot(index="experiment", columns="model", values="f1")
    return means, int((f1["EIF"] > f1["IF"]).sum())


def flag_all_reference(df: pd.DataFrame, n_train: int = 400) -> pd.Series:
    """Mean metrics of a trivial detector that flags every evaluated row as anomalous.

    Over half of the rows after the normal prefix are anomalous, so this
    reference has a high F1 and any useful detector must beat it.
    """
    rates = np.array(
        [
            g["anomaly"].to_numpy()[train_end_index(g["anomaly"], n_train) :].mean()
            for _, g in df.groupby("experiment", sort=True)
        ]
    )
    f1 = 2 * rates / (1 + rates)
    return pd.Series(
        {"precision": rates.mean(), "recall": 1.0, "f1": f1.mean(), "roc_auc": 0.5}
    )
