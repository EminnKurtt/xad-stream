"""Print the README results tables (markdown) from the CSV files in docs/.

`scripts/check_project.sh` checks that every printed line appears verbatim in
README.md, so the README numbers cannot drift from the CSVs.
"""

import pandas as pd

METRICS = ["precision", "recall", "f1", "roc_auc"]
HEADER = "| {} | precision | recall | F1 | ROC AUC |\n|---|---|---|---|---|"


def row(name, values):
    return f"| {name} | " + " | ".join(f"{values[m]:.3f}" for m in METRICS) + " |"


summary = pd.read_csv("docs/compare_if_vs_eif_summary.csv", index_col="model")
print(HEADER.format("model"))
for key, name in {
    "IF": "Isolation Forest",
    "EIF": "Extended Isolation Forest",
    "flag-all": "flag every row (reference)",
}.items():
    print(row(name, summary.loc[key]))

f1 = pd.read_csv("docs/compare_if_vs_eif.csv").pivot(
    index="experiment", columns="model", values="f1"
)
wins = int((f1["EIF"] > f1["IF"]).sum())
print(
    f"\nEIF has a higher F1 than Isolation Forest in {wins} of {len(f1)} experiments.\n"
)

print(HEADER.format("Isolation Forest variant (0.99 quantile)"))
for key, name in {
    "raw": "8 sensors, threshold on training scores",
    "raw_calib": "8 sensors, threshold on held-out 25 %",
    "rolling30": "8 sensors + 30 s rolling mean/std",
}.items():
    print(row(name, pd.read_csv(f"docs/baseline_{key}.csv")[METRICS].mean()))
