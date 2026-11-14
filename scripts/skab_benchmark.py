"""SKAB-protocol benchmark: choose settings on valve1, report on held-out experiments.

Writes docs/skab_grid_dev.csv (every setting tried, on the valve1 experiments),
docs/skab_final_runs.csv (per seed) and docs/skab_final.csv (mean over seeds).
"""

import itertools

import pandas as pd

from xad.benchmark import (
    FEATURES,
    MODELS,
    N_TRAIN,
    QUANTILES,
    SMOOTH,
    evaluate,
    pooled_metrics,
    raw_scores,
    split_experiments,
)
from xad.data import load_skab

SEEDS = range(5)

df = load_skab("data/raw/SKAB")
groups = dict(tuple(df.groupby("experiment", sort=True)))
labels = {e: g["anomaly"].to_numpy() for e, g in groups.items()}
dev, test = split_experiments(groups)
splits = {"held-out (valve2 + other)": test, "all 34": sorted(groups)}
cache = {}


def scores(model, kind, seed):
    if (model, kind, seed) not in cache:
        cache[model, kind, seed] = {
            e: raw_scores(g, model, kind, seed) for e, g in groups.items()
        }
    return cache[model, kind, seed]


# 1. every setting on the development experiments (valve1), seed 0
grid = pd.DataFrame(
    {
        "model": m,
        "features": k,
        "smooth": s,
        "quantile": q,
        **evaluate(scores(m, k, 0), labels, dev, s, q),
    }
    for m, k, s, q in itertools.product(MODELS, FEATURES, SMOOTH, QUANTILES)
).sort_values("f1", ascending=False, kind="stable")
grid.to_csv("docs/skab_grid_dev.csv", index=False, float_format="%.4f")

# 2. per model: the default setting and the one with the best F1 on valve1
configs = {}
for m in MODELS:
    configs[f"{m} default"] = (m, "raw", 1, 0.95)
    best = grid[grid["model"] == m].iloc[0]
    configs[f"{m} selected"] = (
        m,
        best["features"],
        int(best["smooth"]),
        best["quantile"],
    )

# 3. those settings on the held-out experiments and on all 34, over several seeds
runs = pd.DataFrame(
    {
        "config": name,
        "model": m,
        "features": k,
        "smooth": s,
        "quantile": q,
        "split": split,
        "seed": seed,
        **evaluate(scores(m, k, seed), labels, exps, s, q),
    }
    for name, (m, k, s, q) in configs.items()
    for seed in SEEDS
    for split, exps in splits.items()
)
runs.to_csv("docs/skab_final_runs.csv", index=False, float_format="%.4f")
final = runs.groupby(["split", "config"], sort=False).agg(
    features=("features", "first"),
    smooth=("smooth", "first"),
    quantile=("quantile", "first"),
    f1=("f1", "mean"),
    f1_min=("f1", "min"),
    f1_max=("f1", "max"),
    far=("far", "mean"),
    mar=("mar", "mean"),
)
for split, exps in splits.items():  # reference: alarm on every test row
    ref = pooled_metrics((labels[e][N_TRAIN:], labels[e][N_TRAIN:] >= 0) for e in exps)
    final.loc[(split, "flag every row"), ["f1", "f1_min", "f1_max", "far", "mar"]] = (
        ref["f1"],
        ref["f1"],
        ref["f1"],
        ref["far"],
        ref["mar"],
    )
final = final.sort_index(level="split", sort_remaining=False)
final.to_csv("docs/skab_final.csv", float_format="%.4f")

print(f"development: {len(dev)} valve1 experiments; held out: {len(test)}")
print("\n== top 10 settings on valve1 (seed 0) ==")
print(grid.head(10).round(3).to_string(index=False))
print(f"\n== final, mean over seeds {list(SEEDS)} ==")
print(final.round(3).to_string())
