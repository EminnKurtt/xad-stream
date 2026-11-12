import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from xad.eif import ExtendedIsolationForest, average_path_length

SHIFTED = [1, 4, 6]
SEEDS = [0, 1, 2, 3, 4]


def _synthetic(seed, n_normal=1000, n_anom=50, d=8, shift=4.0):
    """Normal training rows, plus a test set whose anomalies are shifted in three features."""
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_normal, d))
    X_test = rng.normal(size=(n_normal + n_anom, d))
    X_test[n_normal:, SHIFTED] += shift
    y_test = np.r_[np.zeros(n_normal), np.ones(n_anom)]
    return X_train, X_test, y_test


def test_average_path_length():
    assert average_path_length(256) == pytest.approx(10.24, abs=0.01)
    assert average_path_length(2) == 1.0
    assert average_path_length(1) == 0.0
    assert average_path_length(np.array([0, 1, 2])).tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("seed", SEEDS)
def test_detects_shift(seed):
    X_train, X_test, y_test = _synthetic(seed)
    model = ExtendedIsolationForest(n_trees=100, random_state=seed).fit(X_train)
    score = model.score_samples(X_test)
    assert ((score > 0) & (score < 1)).all()
    assert roc_auc_score(y_test, score) > 0.95


@pytest.mark.parametrize("seed", SEEDS)
def test_explain_points_to_shifted_features(seed):
    X_train, X_test, y_test = _synthetic(seed)
    model = ExtendedIsolationForest(n_trees=100, random_state=seed).fit(X_train)
    importance = model.explain(X_test)
    assert importance.shape == X_test.shape
    assert (importance >= 0).all()
    top3 = np.argsort(importance[y_test == 1].mean(axis=0))[-3:]
    assert sorted(top3) == SHIFTED


@pytest.mark.parametrize("extension_level", [0, 3, 7])
def test_all_extension_levels_work(extension_level):
    X_train, X_test, y_test = _synthetic(0)
    model = ExtendedIsolationForest(extension_level=extension_level, random_state=0)
    score = model.fit(X_train).score_samples(X_test)
    assert roc_auc_score(y_test, score) > 0.95


def test_deterministic_with_random_state():
    X_train, X_test, _ = _synthetic(0)
    a = ExtendedIsolationForest(n_trees=20, random_state=7).fit(X_train)
    b = ExtendedIsolationForest(n_trees=20, random_state=7).fit(X_train)
    np.testing.assert_array_equal(a.score_samples(X_test), b.score_samples(X_test))
    np.testing.assert_array_equal(a.explain(X_test), b.explain(X_test))


def test_small_and_constant_inputs():
    X = np.ones((10, 3))
    model = ExtendedIsolationForest(n_trees=5, random_state=0).fit(X)
    assert model.sample_size_ == 10
    score = model.score_samples(X)
    assert np.allclose(score, score[0])
    assert (model.explain(X) == 0).all()


def test_invalid_inputs():
    X = np.random.default_rng(0).normal(size=(50, 4))
    with pytest.raises(ValueError):
        ExtendedIsolationForest(extension_level=4).fit(X)
    with pytest.raises(ValueError):
        ExtendedIsolationForest().fit(np.r_[X, [[np.nan] * 4]])
    with pytest.raises(RuntimeError):
        ExtendedIsolationForest().score_samples(X)
    model = ExtendedIsolationForest(n_trees=5).fit(X)
    with pytest.raises(ValueError):
        model.score_samples(X[:, :3])
