"""Print the README results table and its key sentences from the CSV files in docs/.

`scripts/check_project.sh` checks that every printed line appears verbatim in
README.md, so the README numbers cannot drift from the CSVs.
"""

import pandas as pd

OURS = {
    "EIF selected": "Extended Isolation Forest, tuned on valve1",
    "EIF default raw": "Extended Isolation Forest, default",
    "EIF default no_thermo": "Extended Isolation Forest, without Thermocouple",
    "IF selected": "Isolation Forest, tuned on valve1",
    "IF default raw": "Isolation Forest, default",
    "flag every row": "flag every row (trivial reference)",
}

final = pd.read_csv("docs/skab_final.csv").set_index(["split", "config"])
board = pd.read_csv("docs/skab_leaderboard.csv")
board = board[~board["algorithm"].isin(["Perfect detector", "Null detector"])]

rows = [(a, "SKAB leaderboard", f1, far, mar) for a, f1, far, mar in board.to_numpy()]
for key, name in OURS.items():
    r = final.loc[("all 34", key)]
    rows.append((name, "**this repo**", r["f1"], r["far"], r["mar"]))
rows.sort(key=lambda r: -r[2])

print("| detector | source | F1 | FAR, % | MAR, % |\n|---|---|---|---|---|")
for name, source, f1, far, mar in rows:
    print(f"| {name} | {source} | {f1:.2f} | {far:.2f} | {mar:.2f} |")

flag = final.loc[("all 34", "flag every row"), "f1"]
beaten = int((board["f1"] < flag).sum())
print(
    f"\nFlagging every row scores F1 {flag:.2f}, above {beaten} of the "
    f"{len(board)} published methods."
)
held = final.xs("held-out (valve2 + other)")
print("\n| 18 held-out experiments | F1 | FAR, % | MAR, % |\n|---|---|---|---|")
for key, label in {
    "EIF default raw": "default EIF",
    "EIF selected": "EIF tuned on valve1",
    "EIF default no_thermo": "EIF without Thermocouple",
}.items():
    r = held.loc[key]
    print(f"| {label} | {r['f1']:.3f} | {r['far']:.1f} | {r['mar']:.1f} |")

fa = (
    pd.read_csv("docs/false_alarm_positions.csv")
    .groupby("features")
    .sum(numeric_only=True)
)
share = 100 * fa["false_alarms_after"] / fa["normal_after"]
total = fa[["false_alarms_before", "false_alarms_between", "false_alarms_after"]].sum(
    axis=1
)
print(
    f"\n{100 * fa.loc['raw', 'false_alarms_after'] / total['raw']:.0f} % of the default "
    f"EIF's false alarms fall after the labelled anomaly, where it flags "
    f"{share['raw']:.0f} % of the normal rows; without Thermocouple, "
    f"{share['no_thermo']:.0f} %."
)

xb = pd.read_csv("docs/explanation_benchmark_summary.csv").set_index(["task", "method"])
methods = ["EIF explain (ours)", "Local-DIFFI", "z-score", "random"]
tasks = {
    "shift k=1 m=3": "1 sensor shifted by 3 std",
    "shift k=1 m=5": "1 sensor shifted by 5 std",
    "shift k=3 m=3": "3 sensors shifted by 3 std",
    "swap": "correlated pair broken (k = 2)",
}
print(
    "\n| injected anomaly | "
    + " | ".join(methods)
    + " | flagged by EIF / IF |\n|"
    + "---|" * (len(methods) + 2)
)
for task, label in tasks.items():
    p = [f"{xb.loc[(task, m), 'precision']:.2f}" for m in methods]
    eif_det = xb.loc[(task, "EIF explain (ours)"), "detection_rate"]
    if_det = xb.loc[(task, "Local-DIFFI"), "detection_rate"]
    print(f"| {label} | " + " | ".join(p) + f" | {eif_det:.0%} / {if_det:.0%} |")

ev = pd.read_csv("docs/event_metrics_summary.csv", index_col="config")
print(
    "\n| alarms from | anomalies caught | median delay, s "
    "| false-alarm events per hour (before / after the anomaly) |\n|---|---|---|---|"
)
for config, r in ev.iterrows():
    print(
        f"| {config} | {r['detected_pct']:.0f} % | {r['median_delay_s']:.1f} "
        f"| {r['false_events_per_h_before']:.1f} / {r['false_events_per_h_after']:.1f} |"
    )
