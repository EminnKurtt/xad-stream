"""Sensor-level explanation figures and tables for the EIF detector on SKAB."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xad.data import SENSOR_COLUMNS, load_skab
from xad.explain import anomaly_windows, explain_experiment, sensor_summary

EXAMPLE = "valve1/0"
DPI = 100


def shade(ax, labels, end):
    for start, stop in anomaly_windows(labels):
        ax.axvspan(start, stop, color="tab:red", alpha=0.15, lw=0)
    ax.axvline(end, color="grey", ls="--", lw=1)


def plot_example(df_exp, run, importance, path):
    t = np.arange(len(df_exp))
    fig, axes = plt.subplots(
        3, 2, figsize=(10, 7), sharex="col", gridspec_kw={"width_ratios": [40, 1]}
    )
    ax_score, ax_heat, ax_temp = axes[:, 0]
    axes[0, 1].axis("off")
    axes[2, 1].axis("off")

    ax_score.plot(t, run.score, lw=0.8, color="black")
    ax_score.axhline(run.threshold, color="tab:blue", lw=1, label="threshold")
    shade(ax_score, run.y, run.end)
    ax_score.set_ylabel("EIF score")
    ax_score.legend(loc="upper left", fontsize=8)
    ax_score.set_title(
        f"SKAB {EXAMPLE}: red = labelled anomaly, dashed = end of normal prefix",
        fontsize=10,
    )

    im = ax_heat.imshow(
        importance.T,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        extent=(-0.5, len(t) - 0.5, len(SENSOR_COLUMNS) - 0.5, -0.5),
    )
    ax_heat.set_yticks(range(len(SENSOR_COLUMNS)), SENSOR_COLUMNS, fontsize=8)
    ax_heat.axvline(run.end, color="white", ls="--", lw=1)
    fig.colorbar(im, cax=axes[1, 1], label="importance")

    ax_temp.plot(t, df_exp["Temperature"].to_numpy(), lw=0.8, color="black")
    shade(ax_temp, run.y, run.end)
    ax_temp.set_ylabel("Temperature")
    ax_temp.set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_summary(mean, path):
    order = mean.sort_values("share_in").index
    y = np.arange(len(order))
    fig, (ax_imp, ax_shift) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax_imp.barh(
        y + 0.2, mean.loc[order, "share_in"], 0.4, label="inside anomaly windows"
    )
    ax_imp.barh(y - 0.2, mean.loc[order, "share_out"], 0.4, label="outside")
    ax_imp.set_yticks(y, order)
    ax_imp.set_xlabel("mean share of EIF importance (each row sums to 1)")
    ax_imp.legend(fontsize=8, loc="lower right")
    ax_imp.set_title("Explanation (EIF explain)", fontsize=10)
    ax_shift.barh(y, mean.loc[order, "shift"], 0.6, color="grey")
    ax_shift.set_xlabel("|mean inside - mean outside| / std")
    ax_shift.set_title("Visible change in the raw data", fontsize=10)
    fig.suptitle("SKAB, 34 experiments, rows after the normal prefix", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def top3(series):
    return series.nlargest(3).index


df = load_skab("data/raw/SKAB")
summaries, per_exp = [], []
for name, group in df.groupby("experiment", sort=True):
    run, importance = explain_experiment(group)
    if name == EXAMPLE:
        plot_example(group, run, importance, "docs/explanation_valve1_0.png")
    s = sensor_summary(group, run, importance)
    summaries.append(s)
    top_share, top_shift = top3(s["share_in"]), top3(s["shift"])
    per_exp.append(
        {
            "experiment": name,
            "top3_share_in": "|".join(top_share),
            "top3_shift": "|".join(top_shift),
            "overlap": len(set(top_share) & set(top_shift)),
        }
    )

mean = pd.concat(summaries).groupby("sensor").mean().loc[SENSOR_COLUMNS]
mean["share_ratio_in_out"] = mean["share_in"] / mean["share_out"]
mean.to_csv("docs/explanation_summary.csv")
per_exp = pd.DataFrame(per_exp)
per_exp.to_csv("docs/explanation_per_experiment.csv", index=False)
plot_summary(mean, "docs/explanation_summary.png")

print("== mean over experiments, evaluated rows ==")
print(mean.sort_values("share_in", ascending=False).round(3).to_string())
print("\ntop-3 overlap (share inside windows vs visible shift): experiments per value")
print(per_exp["overlap"].value_counts().sort_index().to_string())
chance = 3 * 3 / len(SENSOR_COLUMNS)
print(
    f"mean overlap {per_exp['overlap'].mean():.2f} (two random top-3 sets: {chance:.3f})"
)
