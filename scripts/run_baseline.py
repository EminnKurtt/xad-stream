import pandas as pd

from xad.baseline import evaluate_all
from xad.data import load_skab

df = load_skab("data/raw/SKAB")
summary = {}
CONFIGS = {
    "raw": {"rolling": 0},
    "raw_calib": {"rolling": 0, "calib_frac": 0.25},
    "rolling30": {"rolling": 30},
}
for name, kwargs in CONFIGS.items():
    res = evaluate_all(df, **kwargs)
    res.to_csv(f"docs/baseline_{name}.csv")
    summary[name] = res[["precision", "recall", "f1", "roc_auc"]].mean()
    print(f"\n== {name}: per experiment ==")
    print(res.round(3).to_string())
print("\n== mean over experiments ==")
print(pd.DataFrame(summary).round(3))
