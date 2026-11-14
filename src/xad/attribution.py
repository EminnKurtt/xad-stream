"""Per-sensor attribution methods and a synthetic ground-truth test for them.

Explanations of an anomaly score have no ground truth on real data, so they are
tested on normal rows with injected anomalies whose responsible sensors are known:

- `inject_shift`: k sensors shifted by +-`magnitude` standard deviations. A
  marginal deviation, which a per-sensor z-score explains by construction.
- `inject_swap`: for a correlated pair (a, b), b takes its value from a normal
  row at the opposite end of a. Each sensor stays within its normal range; only
  the relation between them breaks, so a marginal explanation has no advantage.

Methods compared:
- `local_diffi`: Local-DIFFI (Carletti, Terzi, Susto, arXiv:2007.11117,
  Algorithm 3) for scikit-learn's Isolation Forest.
- `ExtendedIsolationForest.explain` (xad.eif): the ExIFFI-inspired heuristic.
- `zscore_importance`: |x| of standardised rows, a model-free baseline.
- random ranking, the chance level.
"""

from __future__ import annotations

from math import ceil, log2

import numpy as np
from sklearn.ensemble import IsolationForest


def local_diffi(model: IsolationForest, X) -> np.ndarray:
    """Local-DIFFI importance for every row of `X`, shape `(n_samples, n_features)`.

    For each tree and each split on the row's path, the split feature gets
    `1/h_t(x) - 1/h_max` (`h_t` = depth of the row's leaf, `h_max` = the tree
    height limit) and its counter is incremented; importance = sum / count.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    h_max = ceil(log2(max(model.max_samples_, 2)))
    total = np.zeros((n, p))
    count = np.zeros((n, p))
    for tree, features in zip(model.estimators_, model.estimators_features_):
        path = tree.decision_path(X[:, features])  # (n, n_nodes) sparse indicator
        depth = np.asarray(path.sum(axis=1)).ravel() - 1
        node_feature = tree.tree_.feature
        internal = node_feature >= 0
        one_hot = np.zeros((len(node_feature), p))
        one_hot[internal, np.asarray(features)[node_feature[internal]]] = 1.0
        uses = path @ one_hot  # times each feature splits on the row's path
        delta = np.divide(1.0, depth, out=np.zeros(n), where=depth > 0) - 1.0 / h_max
        count += uses
        total += uses * delta[:, None]
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def zscore_importance(X_scaled) -> np.ndarray:
    """Absolute standardised deviation of each sensor (standardised on normal rows)."""
    return np.abs(np.asarray(X_scaled, dtype=float))


def precision_at_k(importance: np.ndarray, truth: list[set[int]]) -> np.ndarray:
    """Per row: share of the `len(truth)` top-ranked sensors that are truly responsible.

    Ties are broken towards a fixed but arbitrary order (stable argsort), so a
    constant importance does not get credit for free.
    """
    out = np.empty(len(truth))
    for i, t in enumerate(truth):
        top = np.argsort(-importance[i], kind="stable")[: len(t)]
        out[i] = len(set(top.tolist()) & t) / len(t)
    return out


def inject_shift(
    X: np.ndarray, k: int, magnitude: float, rng
) -> tuple[np.ndarray, list]:
    """Shift `k` random sensors of each standardised row by +-`magnitude`."""
    X = X.copy()
    truth = []
    for row in X:
        sensors = rng.choice(X.shape[1], k, replace=False)
        row[sensors] += magnitude * rng.choice([-1.0, 1.0], k)
        truth.append(set(sensors.tolist()))
    return X, truth


def correlated_pair(X_fit: np.ndarray) -> tuple[int, int, float]:
    """The most strongly correlated sensor pair in the (standardised) fitting rows."""
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(X_fit, rowvar=False))
    np.fill_diagonal(corr, 0.0)
    a, b = np.unravel_index(np.abs(corr).argmax(), corr.shape)
    return int(a), int(b), float(corr[a, b])


def inject_swap(
    X: np.ndarray, X_fit: np.ndarray, a: int, b: int, rng, min_abs: float = 1.0
) -> tuple[np.ndarray, list]:
    """Give b a value from a fitting row where a sits at the opposite end.

    Only rows with `|x_a| >= min_abs` are used (elsewhere the swap changes
    little); donors are fitting rows with `x_a` of opposite sign and
    `|x_a| >= min_abs`. Returns the modified rows and their truth `{a, b}`.
    """
    rows = X[np.abs(X[:, a]) >= min_abs].copy()
    pos = X_fit[X_fit[:, a] >= min_abs, b]
    neg = X_fit[X_fit[:, a] <= -min_abs, b]
    keep = []
    for i, row in enumerate(rows):
        donors = neg if row[a] > 0 else pos
        if donors.size:
            row[b] = rng.choice(donors)
            keep.append(i)
    return rows[keep], [{a, b}] * len(keep)
