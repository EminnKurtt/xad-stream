# Results

All numbers below come from the CSV files in this folder. Regenerate them with
`python scripts/skab_benchmark.py`, `python scripts/compare_models.py`,
`python scripts/run_baseline.py` and `python scripts/explain_skab.py`.

## Main result: SKAB benchmark protocol

`python scripts/skab_benchmark.py` follows SKAB's outlier-detection protocol. In each
experiment the first 400 rows are for training and the rest for testing. TP, FP, FN and TN
are pooled over all test rows. On the same predictions, `xad.benchmark.pooled_metrics`
returns exactly the values of SKAB's own `core/metrics.py` `chp_score(metric="binary")`.
Inside the 400 training rows, models fit on the first 300, and the threshold is a quantile
of the scores on the last 100.

**Selection.** All 96 settings were evaluated on the 16 valve1 experiments only, with seed 0
(`skab_grid_dev.csv`). The settings tried were:
- model: IF or EIF;
- features: raw, linearly detrended on the fitting rows, without Thermocouple, or
  Thermocouple replaced by its change over 30 rows;
- causal score smoothing: 1, 5, 15 or 31 rows;
- threshold quantile: 0.95, 0.99 or 1.0.

For each model, the setting with the best valve1 F1 was kept ("selected"). Each feature set
except detrending was also run with the default post-processing (no smoothing, q 0.95).

**Evaluation.** Those settings were then run with seeds 0–4 (`skab_final_runs.csv`, means in
`skab_final.csv`):

| split | detector | F1 (min–max over seeds) | FAR, % | MAR, % |
|---|---|---|---|---|
| held-out (valve2 + other) | EIF default, raw | 0.777 (0.777–0.778) | 47.5 | 10.4 |
| held-out (valve2 + other) | EIF selected (raw, smoothing 5, q 1.0) | 0.777 (0.774–0.780) | 52.0 | 7.9 |
| held-out (valve2 + other) | EIF default, without Thermocouple | 0.772 (0.770–0.775) | 34.4 | 18.4 |
| held-out (valve2 + other) | EIF default, Thermocouple 30 s change | 0.772 (0.769–0.776) | 34.8 | 18.2 |
| held-out (valve2 + other) | IF default, raw | 0.687 (0.679–0.698) | 28.5 | 34.9 |
| held-out (valve2 + other) | IF default, without Thermocouple | 0.633 (0.618–0.646) | 20.2 | 45.6 |
| held-out (valve2 + other) | IF selected (raw, smoothing 31, q 0.95) | 0.752 (0.750–0.754) | 71.0 | 2.9 |
| held-out (valve2 + other) | flag every row | 0.698 | 100.0 | 0.0 |
| all 34 | EIF default, raw | 0.768 (0.764–0.770) | 48.1 | 11.8 |
| all 34 | EIF selected | 0.775 (0.772–0.776) | 51.8 | 8.5 |
| all 34 | EIF default, without Thermocouple | 0.759 (0.758–0.760) | 31.5 | 22.1 |
| all 34 | IF default, raw | 0.604 (0.590–0.626) | 23.4 | 48.0 |
| all 34 | IF selected | 0.753 (0.750–0.757) | 62.3 | 7.1 |
| all 34 | flag every row | 0.698 | 100.0 | 0.0 |

Findings:

- **Most false alarms come after the anomaly.** Default EIF, seed 0
  (`false_alarm_positions.csv`):
  - It flags 1,479 of the 5,769 normal test rows before the labelled anomaly (26 %) and
    3,890 of the 5,261 after it (74 %). So 72 % of its false alarms fall after the anomaly.
  - SKAB labels those rows normal again, but the plant has not returned to its starting
    state. Averaged over the 32 experiments with at least 30 rows after the anomaly,
    Thermocouple ends 11.5 training-row standard deviations away from its training mean.
    No other sensor ends more than 1.9 away (`post_anomaly_shift.csv`).
- **Dropping Thermocouple trades missed anomalies for fewer false alarms.**
  - It cuts the post-anomaly false-alarm rate from 74 % to 43 %.
  - On the held-out experiments, FAR drops from 47.5 % to 34.4 %, while F1 barely moves
    (0.777 to 0.772) and MAR rises from 10.4 % to 18.4 %.
  - Using Thermocouple's 30 s change instead gives almost the same result.
  - These variants were proposed after the position analysis above, which covered all 34
    experiments. Their settings, though, were not tuned on the held-out ones.
  - For Isolation Forest, dropping Thermocouple lowers F1 (0.687 to 0.633).

- **On F1, EIF is level with the best published methods:** Conv-AE 0.78, MSET 0.78,
  T-squared+Q 0.76 (`skab_leaderboard.csv`, quoted from the SKAB README). Its false-alarm
  rate, though, is higher than all four top methods: 48 % against 13.55–39.73 %. EIF
  reaches its F1 through recall, not precision.
- **Tuning on valve1 did not improve the held-out F1** (0.777 for both), and it raised the
  false-alarm rate. For Isolation Forest, tuning chose heavy smoothing. That lifted F1 from
  0.687 to 0.752 on the held-out experiments, but it pushed FAR to 71 %.
- **Linear detrending failed.** The best detrended setting on valve1 had
  F1 0.742 (EIF) and 0.744 (IF), against 0.773 and 0.755 raw, and its FAR rose above 70 %.
  Projecting a straight line fitted on 300 rows over the roughly 700 test rows
  overcorrects.
- **F1 on SKAB is dominated by how many rows are anomalous.** Flagging every row scores
  0.698, above 6 of the 10 published methods (T-squared, LSTM-VAE, Vanilla LSTM, MSCRED,
  Vanilla AE, Isolation forest). Results on SKAB should always report FAR and MAR next
  to F1.
- **The threshold matters as much as the model.** The leaderboard's Isolation Forest scores
  F1 0.29; it uses a fixed contamination of 0.0005, as in SKAB's
  `notebooks/isolation_forest.ipynb`. Our Isolation Forest scores 0.604 with a threshold
  calibrated on held-out training rows.

## Explanations against a known ground truth

`python scripts/explanation_benchmark.py` (methods in `src/xad/attribution.py`).
For each of the 34 experiments and seeds 0–2:

- The scaler, the EIF and the Isolation Forest are fit on rows 0–299, exactly as in the
  SKAB-protocol benchmark.
- Anomalies with known responsible sensors are injected into the 100 normal rows that
  follow.

Injected anomaly types, in standardised units:
- **shift:** k ∈ {1, 2, 3} random sensors move by ±m standard deviations, m ∈ {2, 3, 5}.
- **swap:** for the most correlated sensor pair (a, b) in the fitting rows, used only
  where |r| ≥ 0.6 (14 experiments), rows with |x_a| ≥ 1 get a value of b from a fitting
  row where a sits at the opposite end. b stays within its normal range; only its
  relation to a breaks. The truth is {a, b}.

Methods compared:
- the EIF `explain` heuristic (ours);
- Local-DIFFI on the 200-tree Isolation Forest, from Algorithm 3 of the DIFFI paper. A
  test checks it against a direct tree-walk implementation of that algorithm;
- |z|, the per-sensor deviation from the fitting rows;
- random ranking.

**Metric.** precision@k: the share of the responsible sensors among each method's top k,
where k is the number of responsible sensors. It is averaged over all injected rows,
whether or not the detector flagged them. Experiments are weighted equally, then results
are averaged over seeds (`explanation_benchmark_summary.csv`; per experiment in
`explanation_benchmark.csv`). The spread across seeds is at most 0.061.

| injected anomaly | EIF explain (ours) | Local-DIFFI | z-score | random | flagged by EIF / IF |
|---|---|---|---|---|---|
| shift k=1, m=2 | 0.480 | 0.459 | 0.479 | 0.129 | 21 % / 15 % |
| shift k=1, m=3 | 0.746 | 0.650 | 0.739 | 0.134 | 43 % / 19 % |
| shift k=1, m=5 | 0.967 | 0.773 | 0.966 | 0.125 | 81 % / 24 % |
| shift k=2, m=3 | 0.809 | 0.734 | 0.806 | 0.260 | 71 % / 47 % |
| shift k=3, m=3 | 0.843 | 0.784 | 0.840 | 0.385 | 84 % / 73 % |
| swap (k = 2) | 0.461 | 0.529 | 0.403 | 0.245 | 14 % / 11 % |

Findings:

- **On shifted sensors our explanation matches the z-score to within 0.01 in every
  setting.** It adds nothing over the much simpler per-sensor deviation. Local-DIFFI is
  0.02–0.19 lower. Every method is far above chance.
- **On broken pairs the order reverses:** Local-DIFFI 0.529, ours 0.461, z-score 0.403,
  chance 0.245. The model-based explanations do capture something about relations between
  sensors. But both detectors flag only 11–14 % of these rows, because in SKAB's normal
  operation the strongest sensor pair has a median |r| of about 0.6.
- **The EIF detects injected shifts much more often than the Isolation Forest**
  (1 sensor at 5 std: 81 % against 24 %), both with thresholds calibrated the same way.
- **Local-DIFFI needs many trees.** On synthetic data with 3 shifted features its top-3
  precision was 0.68–0.75 with 50 trees and 0.94–0.98 with 200 trees (development check,
  not saved). The benchmark uses 200.
- **Limits:** the injected anomalies are two simple types placed on real normal rows; real
  faults may look different. "Responsible sensor" is defined by the injection, which for
  swaps includes both sensors of the pair.

## Per-experiment protocol (earlier analysis)

The sections below use the earlier protocol. The normal prefix ends 5 rows before the first
labelled anomaly, which uses the labels, and metrics are averaged per experiment rather
than pooled. The numbers are therefore not comparable with the SKAB leaderboard.

### Isolation Forest vs Extended Isolation Forest

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

### Isolation Forest variants (0.99 quantile)

Means of `baseline_raw.csv`, `baseline_raw_calib.csv` and `baseline_rolling30.csv`:

| Isolation Forest variant | precision | recall | F1 | ROC AUC |
|---|---|---|---|---|
| 8 sensors, threshold on training scores | 0.740 | 0.419 | 0.498 | 0.742 |
| 8 sensors, threshold on held-out 25 % | 0.720 | 0.316 | 0.397 | 0.732 |
| 8 sensors + 30 s rolling mean/std | 0.603 | 0.346 | 0.355 | 0.733 |

Rolling-window features did not help, and neither did a held-out threshold at this quantile.

## Do the explanations point to the right sensors?

(The explanations use the per-experiment protocol above.)

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
