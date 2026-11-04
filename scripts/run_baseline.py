import pandas as pd

from xad.baseline import evaluate_all
from xad.data import load_skab

df = load_skab("data/raw/SKAB")
summary = {}
for name, rolling in {"raw": 0, "rolling30": 30}.items():
    res = evaluate_all(df, rolling=rolling)
    res.to_csv(f"docs/baseline_{name}.csv")
    summary[name] = res[["precision", "recall", "f1", "roc_auc"]].mean()
    print(f"\n== {name}: per experiment ==")
    print(res.round(3).to_string())
print("\n== mean over experiments ==")
print(pd.DataFrame(summary).round(3))
