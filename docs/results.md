# Results

All numbers below come from the CSV files in this folder. Regenerate them with
`python scripts/run_baseline.py`, `python scripts/compare_models.py` and
`python scripts/explain_skab.py`.
Every metric is a mean over the 34 labelled SKAB experiments, computed on the rows after
each experiment's known-normal prefix.

## Isolation Forest vs Extended Isolation Forest

Same protocol for both models. Each is fit on the first 75 % of the normal prefix, and the
threshold is the 0.95 quantile of the scores on the held-out last 25 %
(`compare_if_vs_eif_summary.csv`, per experiment in `compare_if_vs_eif.csv`).

| model | precision | recall | F1 | ROC AUC |
|---|---|---|---|---|
| Isolation Forest | 0.702 | 0.521 | 0.563 | 0.732 |
| Extended Isolation Forest | 0.696 | 0.890 | 0.770 | 0.796 |
| flag every row (reference) | 0.542 | 1.000 | 0.702 | 0.500 |

EIF has a higher F1 than Isolation Forest in 29 of the 34 experiments. Read the F1 values
against the reference row: more than half of the evaluated rows are anomalous, so
flagging every row already gives F1 0.702. EIF beats that reference and Isolation Forest
does not. Most of EIF's F1 gain comes from recall: with the same quantile, its
threshold flags many more rows. The threshold-free ROC AUC improves less, from 0.732 to 0.796.

## Do the explanations point to the right sensors?

`explain` is a simplified heuristic inspired by ExIFFI, not ExIFFI itself (see
`src/xad/eif.py`). Raw importance is smaller for rows that are easy to isolate,
because their paths are shorter and collect fewer splits. For that reason the comparison
uses each row's *share* of importance (shares sum to 1 per row). As a model-free reference, `shift` measures
how visibly each sensor changes: `|mean inside - mean outside| / std` of the raw
values (`explanation_summary.csv`, `explanation_summary.png`).

| sensor | share inside windows | share outside | ratio | visible shift |
|---|---|---|---|---|
| Volume Flow RateRMS | 0.217 | 0.110 | 1.973 | 1.302 |
| Thermocouple | 0.188 | 0.216 | 0.869 | 0.418 |
| Temperature | 0.141 | 0.169 | 0.835 | 0.500 |
| Accelerometer1RMS | 0.128 | 0.108 | 1.189 | 0.453 |
| Accelerometer2RMS | 0.128 | 0.106 | 1.206 | 0.556 |
| Pressure | 0.070 | 0.102 | 0.685 | 0.065 |
| Voltage | 0.066 | 0.097 | 0.680 | 0.060 |
| Current | 0.064 | 0.094 | 0.680 | 0.099 |

**Partly yes.**

- **Volume Flow RateRMS behaves as hoped.** It changes most visibly, and its share of
  importance almost doubles inside the anomaly windows.
- **Pressure, Voltage and Current are ranked last,** and they barely change.
- **Thermocouple and Temperature are only half a success.** They have the two largest
  shares overall, but their share is *lower* inside the windows than outside. Both drift
  slowly after the short normal prefix, so the model already attributes normal rows to
  them. The explanation names them, but it does not single out the anomaly windows with
  them. `explanation_valve1_0.png` shows this clearly. Temperature importance lights up
  during the labelled temperature drop. But Thermocouple importance is already high from
  the end of the prefix onwards, before any anomaly, and the score drifts towards the
  threshold with it.
- **The accelerometers change about as much as Temperature** by this measure, and their
  share rises slightly inside the windows. So "vibration barely changes" holds only for
  this crude mean-difference test.

Per experiment (`explanation_per_experiment.csv`), the top-3 sensors by share inside the
windows overlap the top-3 by visible shift in 2.15 of 3 positions on average. Two random
top-3 sets would overlap in 1.125. All three match in 11 experiments, and no experiment
has zero overlap.

Limitations of this check:

- `shift` is a univariate mean difference, so it misses changes in variance or in the
  relation between sensors.
- The explanation inherits the model's sensitivity to drift outside the training prefix.
