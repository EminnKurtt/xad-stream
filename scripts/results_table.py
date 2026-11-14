"""Print the README results table and its key sentences from the CSV files in docs/.

`scripts/check_project.sh` checks that every printed line appears verbatim in
README.md, so the README numbers cannot drift from the CSVs.
"""

import pandas as pd

OURS = {
    "EIF selected": "Extended Isolation Forest, tuned on valve1",
    "EIF default": "Extended Isolation Forest, default",
    "IF selected": "Isolation Forest, tuned on valve1",
    "IF default": "Isolation Forest, default",
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
d, s = held.loc["EIF default"], held.loc["EIF selected"]
print(
    f"On the 18 held-out experiments the default EIF scores F1 {d['f1']:.3f} "
    f"(FAR {d['far']:.1f} %), and the setting tuned on valve1 F1 {s['f1']:.3f} "
    f"(FAR {s['far']:.1f} %)."
)
