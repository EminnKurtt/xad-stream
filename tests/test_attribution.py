from math import ceil, log2

import numpy as np
from sklearn.ensemble import IsolationForest

from xad.attribution import (
    correlated_pair,
    inject_shift,
    inject_swap,
    local_diffi,
    precision_at_k,
    zscore_importance,
)

SHIFTED = [1, 4, 6]


def _reference_local_diffi(model, x):
    """Algorithm 3 of the DIFFI paper, one row, walking each tree explicitly."""
    p = len(x)
    h_max = ceil(log2(model.max_samples_))
    total, count = np.zeros(p), np.zeros(p)
    for tree, features in zip(model.estimators_, model.estimators_features_):
        t = tree.tree_
        node, path = 0, []
        while t.children_left[node] != -1:
            f = features[t.feature[node]]
            path.append(f)
            go_left = x[f] <= t.threshold[node]
            node = t.children_left[node] if go_left else t.children_right[node]
        for f in path:
            count[f] += 1
            total[f] += 1 / len(path) - 1 / h_max
    return np.divide(total, count, out=np.zeros(p), where=count > 0)


def _model_and_rows(seed=0):
    rng = np.random.default_rng(seed)
    # Local-DIFFI is noisy with few trees; 200 matches the benchmark
    model = IsolationForest(n_estimators=200, random_state=seed).fit(
        rng.normal(size=(500, 8))
    )
    rows = rng.normal(size=(40, 8))
    rows[:20, SHIFTED] += 4.0
    return model, rows


def test_local_diffi_matches_reference_algorithm():
    model, rows = _model_and_rows()
    fast = local_diffi(model, rows)
    slow = np.array([_reference_local_diffi(model, x) for x in rows])
    np.testing.assert_allclose(fast, slow, atol=1e-12)


def test_local_diffi_points_to_shifted_features():
    model, rows = _model_and_rows()
    importance = local_diffi(model, rows[:20])
    assert precision_at_k(importance, [set(SHIFTED)] * 20).mean() > 0.9


def test_precision_at_k_and_zscore():
    importance = np.array([[0.1, 0.9, 0.5], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(precision_at_k(importance, [{1, 2}, {2}]), [1.0, 0.0])
    np.testing.assert_allclose(zscore_importance([[-2.0, 1.0]]), [[2.0, 1.0]])


def test_inject_shift_changes_only_the_true_sensors():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    shifted, truth = inject_shift(X, k=2, magnitude=3.0, rng=rng)
    for orig, new, t in zip(X, shifted, truth):
        changed = set(np.flatnonzero(~np.isclose(orig, new)).tolist())
        assert changed == t and len(t) == 2
        np.testing.assert_allclose(np.abs(new - orig)[list(t)], 3.0)


def test_inject_swap_breaks_the_relation_only():
    rng = np.random.default_rng(0)
    a_vals = rng.normal(size=600)
    X_all = np.c_[a_vals, a_vals + 0.3 * rng.normal(size=600), rng.normal(size=600)]
    X_fit, X = X_all[:300], X_all[300:]
    a, b, corr = correlated_pair(X_fit)
    assert {a, b} == {0, 1} and corr > 0.9
    swapped, truth = inject_swap(X, X_fit, a, b, rng)
    assert len(swapped) > 0 and truth[0] == {a, b}
    assert (np.abs(swapped[:, a]) >= 1.0).all()
    # b now sits on the opposite side of a: the relation is broken
    assert (np.sign(swapped[:, a]) != np.sign(swapped[:, b])).mean() > 0.9
    # each value is still within the normal range of its sensor
    assert np.abs(swapped[:, b]).max() <= np.abs(X_fit[:, b]).max()
