"""Extended Isolation Forest (EIF) with a simple per-feature explanation.

The model follows Hariri, Kind and Brunner, "Extended Isolation Forest",
IEEE TKDE 2019: every split is a random hyperplane instead of an axis-parallel
cut. `extension_level = 0` gives axis-parallel splits (like the original
Isolation Forest); `extension_level = d - 1` gives fully random hyperplanes.

`ExtendedIsolationForest.explain` is *inspired by* ExIFFI, but it is not a
reimplementation of it. It is a simplified heuristic: at every split on a
sample's path, each feature gets its share of the split value
`|n_k (x_k - p_k)| / sum_j |n_j (x_j - p_j)|`, weighted by how much the split
isolates the sample, `1 - size(child) / size(node)`. Shares are summed along the
path and averaged over trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import numpy as np


def average_path_length(n):
    """Average path length `c(n)` of an unsuccessful search in a binary tree.

    `c(n) = 2 (ln(n - 1) + gamma) - 2 (n - 1) / n` for `n > 2`, with the usual
    conventions `c(2) = 1` and `c(n) = 0` for `n <= 1`.
    """
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    big = n > 2
    out[big] = (
        2.0 * (np.log(n[big] - 1.0) + np.euler_gamma) - 2.0 * (n[big] - 1.0) / n[big]
    )
    return out if out.ndim else float(out)


@dataclass
class _Tree:
    """One isolation tree stored as flat arrays; `left == -1` marks a leaf."""

    normal: np.ndarray  # (n_nodes, d) hyperplane normal vectors
    point: np.ndarray  # (n_nodes, d) hyperplane intercept points
    left: np.ndarray  # (n_nodes,) child index, -1 for leaves
    right: np.ndarray  # (n_nodes,)
    size: np.ndarray  # (n_nodes,) training samples that reached the node


def _build_tree(X: np.ndarray, height_limit: int, n_nonzero: int, rng) -> _Tree:
    d = X.shape[1]
    normal, point, left, right, size = [], [], [], [], []

    def new_node(n_rows: int) -> int:
        normal.append(np.zeros(d))
        point.append(np.zeros(d))
        left.append(-1)
        right.append(-1)
        size.append(n_rows)
        return len(size) - 1

    stack = [(new_node(len(X)), X, 0)]
    while stack:
        node, Xn, depth = stack.pop()
        if depth >= height_limit or len(Xn) <= 1:
            continue
        lo, hi = Xn.min(axis=0), Xn.max(axis=0)
        if np.array_equal(lo, hi):  # identical rows cannot be separated
            continue
        n = np.zeros(d)
        n[rng.choice(d, n_nonzero, replace=False)] = rng.standard_normal(n_nonzero)
        p = rng.uniform(lo, hi)
        go_left = ((Xn - p) * n).sum(axis=1) <= 0
        n_left = int(go_left.sum())
        lc, rc = new_node(n_left), new_node(len(Xn) - n_left)
        normal[node], point[node], left[node], right[node] = n, p, lc, rc
        stack.append((lc, Xn[go_left], depth + 1))
        stack.append((rc, Xn[~go_left], depth + 1))

    return _Tree(
        np.array(normal),
        np.array(point),
        np.array(left),
        np.array(right),
        np.array(size, float),
    )


class ExtendedIsolationForest:
    """Extended Isolation Forest (Hariri et al., 2019) with `explain`.

    Parameters
    ----------
    n_trees: number of isolation trees.
    sample_size: rows drawn without replacement for each tree (capped at the
        number of training rows).
    extension_level: number of extra non-zero coordinates of each hyperplane
        normal, in `[0, d - 1]`; `None` means `d - 1` (fully extended).
    random_state: seed for `numpy.random.default_rng`.
    """

    def __init__(
        self,
        n_trees: int = 100,
        sample_size: int = 256,
        extension_level: int | None = None,
        random_state: int | None = None,
    ):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.extension_level = extension_level
        self.random_state = random_state

    def fit(self, X) -> ExtendedIsolationForest:
        X = self._validate(X, fitting=True)
        n, d = X.shape
        if self.n_trees < 1:
            raise ValueError(f"n_trees must be >= 1, got {self.n_trees}")
        if self.sample_size < 2:
            raise ValueError(f"sample_size must be >= 2, got {self.sample_size}")
        ext = d - 1 if self.extension_level is None else self.extension_level
        if not 0 <= ext <= d - 1:
            raise ValueError(f"extension_level must be in [0, {d - 1}], got {ext}")

        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = d
        self.sample_size_ = min(self.sample_size, n)
        self.height_limit_ = ceil(log2(self.sample_size_))
        self.trees_ = [
            _build_tree(
                X[rng.choice(n, self.sample_size_, replace=False)],
                self.height_limit_,
                ext + 1,
                rng,
            )
            for _ in range(self.n_trees)
        ]
        return self

    def score_samples(self, X) -> np.ndarray:
        """Anomaly score `2 ** (-E[h(x)] / c(sample_size))` in (0, 1); higher = more anomalous."""
        X = self._validate(X)
        paths = np.mean(
            [self._traverse(tree, X, explain=False)[0] for tree in self.trees_], axis=0
        )
        return 2.0 ** (-paths / average_path_length(self.sample_size_))

    def explain(self, X) -> np.ndarray:
        """Per-feature importance, shape `(n_samples, n_features)` (ExIFFI-inspired heuristic)."""
        X = self._validate(X)
        return np.mean(
            [self._traverse(tree, X, explain=True)[1] for tree in self.trees_], axis=0
        )

    @staticmethod
    def _traverse(tree: _Tree, X: np.ndarray, explain: bool):
        """Send all rows down one tree level by level; return path lengths and importances."""
        node = np.zeros(len(X), dtype=int)
        depth = np.zeros(len(X))
        importance = np.zeros_like(X) if explain else None
        rows = np.flatnonzero(tree.left[node] >= 0)
        while rows.size:
            nd = node[rows]
            contrib = tree.normal[nd] * (X[rows] - tree.point[nd])
            child = np.where(contrib.sum(axis=1) <= 0, tree.left[nd], tree.right[nd])
            if explain:
                mag = np.abs(contrib)
                total = mag.sum(axis=1, keepdims=True)
                share = np.divide(mag, total, out=np.zeros_like(mag), where=total > 0)
                weight = 1.0 - tree.size[child] / tree.size[nd]
                importance[rows] += share * weight[:, None]
            node[rows] = child
            depth[rows] += 1
            rows = rows[tree.left[child] >= 0]
        return depth + average_path_length(tree.size[node]), importance

    def _validate(self, X, fitting: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        if fitting:
            if len(X) < 2:
                raise ValueError("need at least 2 rows to fit")
        else:
            if not hasattr(self, "trees_"):
                raise RuntimeError("call fit before scoring")
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"expected {self.n_features_in_} features, got {X.shape[1]}"
                )
        return X
