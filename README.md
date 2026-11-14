# XAD-Stream

[![CI](https://github.com/EminnKurtt/xad-stream/actions/workflows/ci.yml/badge.svg)](https://github.com/EminnKurtt/xad-stream/actions/workflows/ci.yml)

Explainable anomaly detection on industrial sensor data, evaluated on the public SKAB benchmark.

Industrial machines are monitored by many sensors, and faults are rare and often unlabelled.
An alarm is more useful when it also says *which sensors* look wrong. XAD-Stream fits
unsupervised isolation-based detectors to normal operation and flags deviations. It also
attributes each anomaly score to individual sensors, checks honestly whether those attributions
point to the sensors that actually change, and replays the data as a stream.

## Results

These results use SKAB's own outlier-detection protocol. In each of the 34 experiments, the
first 400 rows are for training and the rest for testing. Metrics pool all test rows. Our
metric code reproduces SKAB's `chp_score` exactly. Our numbers are means over 5 seeds; the
leaderboard rows are copied from the
[SKAB README](https://github.com/waico/SKAB#outlier-detection-problem).

| detector | source | F1 | FAR, % | MAR, % |
|---|---|---|---|---|
| Conv-AE | SKAB leaderboard | 0.78 | 13.55 | 28.02 |
| MSET | SKAB leaderboard | 0.78 | 39.73 | 14.13 |
| Extended Isolation Forest, tuned on valve1 | **this repo** | 0.77 | 51.78 | 8.51 |
| Extended Isolation Forest, default | **this repo** | 0.77 | 48.13 | 11.83 |
| T-squared+Q (PCA-based) | SKAB leaderboard | 0.76 | 26.62 | 24.92 |
| Extended Isolation Forest, without Thermocouple | **this repo** | 0.76 | 31.53 | 22.15 |
| Isolation Forest, tuned on valve1 | **this repo** | 0.75 | 62.31 | 7.11 |
| LSTM-AE | SKAB leaderboard | 0.74 | 29.96 | 25.92 |
| flag every row (trivial reference) | **this repo** | 0.70 | 100.00 | 0.00 |
| T-squared | SKAB leaderboard | 0.66 | 19.21 | 42.60 |
| Isolation Forest, default | **this repo** | 0.60 | 23.45 | 47.96 |
| LSTM-VAE | SKAB leaderboard | 0.56 | 9.13 | 55.03 |
| Vanilla LSTM | SKAB leaderboard | 0.54 | 12.54 | 59.53 |
| Vanilla AE | SKAB leaderboard | 0.39 | 2.59 | 75.15 |
| MSCRED | SKAB leaderboard | 0.36 | 49.94 | 69.88 |
| Isolation forest | SKAB leaderboard | 0.29 | 2.56 | 82.89 |

FAR is the share of normal rows flagged; MAR is the share of anomalous rows missed.

What this shows:

- **F1 alone is a weak ranking on SKAB.** More than half of the test rows are anomalous.
  Flagging every row scores F1 0.70, above 6 of the 10 published methods.
  FAR and MAR have to be read alongside F1.
- **The Extended Isolation Forest matches the best published F1, but with more false alarms.**
  It is a 200-tree forest fit on 300 rows per experiment, with no deep learning. It misses
  few anomalies, but it flags about half of the normal rows. The best deep model (Conv-AE)
  reaches the same F1 with far fewer false alarms.
- **Most false alarms come after the anomaly, and the explanations found the cause.**
  72 % of the default EIF's false alarms fall after the labelled anomaly, where it flags 74 % of the normal rows; without Thermocouple, 43 %.
  SKAB labels those rows normal again. But the fluid temperature (Thermocouple) has
  settled at a new level, and the explanation method had already singled it out (see
  below). Dropping Thermocouple is a better trade-off: on valve1, no raw-sensor EIF setting
  reaches a false-alarm rate below 35 %, while the EIF without it does (F1 0.757, FAR
  29.7 %).
- **Results on experiments never used for any choice.** Settings were chosen on the 16
  valve1 experiments only. On the 18 held-out ones, tuning the threshold and smoothing did
  not help, and removing a linear drift made results worse. Dropping Thermocouple cut false
  alarms by about a quarter, for slightly lower F1. Every setting tried is listed in
  [`docs/skab_grid_dev.csv`](docs/skab_grid_dev.csv).

| 18 held-out experiments | F1 | FAR, % | MAR, % |
|---|---|---|---|
| default EIF | 0.777 | 47.5 | 10.4 |
| EIF tuned on valve1 | 0.777 | 52.0 | 7.9 |
| EIF without Thermocouple | 0.772 | 34.4 | 18.4 |

- **The threshold matters as much as the model.** The leaderboard's Isolation Forest
  (F1 0.29) and ours (F1 0.60) use the same algorithm. Ours sets its threshold on held-out
  normal rows; theirs uses a fixed contamination value.

Full write-up, including per-experiment results and the explanation analysis:
[`docs/results.md`](docs/results.md).

### Explanations

![EIF score, sensor importance and Temperature for SKAB valve1/0](docs/explanation_valve1_0.png)

The false alarms have a visible cause. In `valve1/0`, Temperature importance rises during
the labelled temperature drop. But Thermocouple importance is already high before the
anomaly: the fluid temperature drifts away from the short training window and pushes the
score over the threshold. The explanation shows *why* the detector raises those false alarms.

![Share of importance per sensor inside and outside anomaly windows](docs/explanation_summary.png)

Across all experiments the result is mixed:

- **Volume Flow RateRMS** changes most visibly, and its share of importance roughly doubles
  inside anomaly windows (0.110 outside, 0.217 inside).
- **Pressure, Voltage and Current** barely change, and they get the lowest share.
- **Temperature and Thermocouple** get high importance everywhere, so their importance does
  not single out the anomaly windows. For Thermocouple this pointed to the real problem:
  the detector raises false alarms on it after the anomaly.

### Do the explanations find the responsible sensor?

Real anomalies have no sensor-level ground truth. So `scripts/explanation_benchmark.py`
injects anomalies with known causes into real SKAB normal rows (100 rows per experiment,
34 experiments, 3 seeds; models fit on the 300 rows before them):

- **Shift:** k sensors are moved by a set number of standard deviations.
- **Broken pair:** for the most correlated sensor pair (|r| ≥ 0.6, 14 experiments), one
  sensor takes its value from a normal row where the other sensor sits at the opposite end.
  Each value is normal on its own; only the relation between them is wrong.

Each cell is precision@k: the share of the truly responsible sensors among an explanation's
top k, where k is the number of responsible sensors. The score covers every injected row,
whether or not the detector raised an alarm.

| injected anomaly | EIF explain (ours) | Local-DIFFI | z-score | random | flagged by EIF / IF |
|---|---|---|---|---|---|
| 1 sensor shifted by 3 std | 0.75 | 0.65 | 0.74 | 0.13 | 43% / 19% |
| 1 sensor shifted by 5 std | 0.97 | 0.77 | 0.97 | 0.13 | 81% / 24% |
| 3 sensors shifted by 3 std | 0.84 | 0.78 | 0.84 | 0.39 | 84% / 73% |
| correlated pair broken (k = 2) | 0.46 | 0.53 | 0.40 | 0.24 | 14% / 11% |

- **On shifted sensors, our EIF explanation is as accurate as a plain z-score,** but not
  more. It ranks the sensors almost exactly like the per-sensor deviation from normal.
  Local-DIFFI on Isolation Forest is less accurate. All three are far above chance.
- **On broken relations, the model-based explanations do better than the z-score**
  (Local-DIFFI 0.53, ours 0.46, z-score 0.40). But both detectors flag only 11–14 % of
  these rows. In SKAB's normal operation the sensors are only moderately correlated, so
  neither model learns much about their relations.
- **What this means in practice:** on SKAB, much of what the explanation adds over a
  per-sensor z-score shows up only for anomalies the detector rarely catches. This
  matches recent work reporting that anomalies in multivariate benchmarks are mostly
  visible in single channels (Pinet et al., see References). Full write-up:
  [`docs/results.md`](docs/results.md).

## Method

- **Protocol.** SKAB's protocol: per experiment, the first 400 rows are for training and
  the rest for testing. No labels are used for training. Inside the training rows, the
  scaler and the model are fit on the first 300, and the alarm threshold is a quantile of
  the scores on the last 100.
- **Models.**
  - **Isolation Forest:** scikit-learn, 200 trees.
  - **Extended Isolation Forest:** implemented from scratch in
    [`src/xad/eif.py`](src/xad/eif.py), 200 trees, fully extended. It splits on random
    hyperplanes instead of single features (Hariri et al.).
- **Tuning, done fairly.** Score smoothing (1, 5, 15 or 31 rows), threshold quantile
  (0.95, 0.99 or 1.0) and features (raw, linearly detrended, without Thermocouple, or
  Thermocouple as its 30 s change) were chosen on the valve1 experiments only. They were
  then evaluated on the held-out valve2 and other experiments. One caveat: the
  Thermocouple variants were proposed after an analysis that looked at all 34 experiments
  (`scripts/false_alarm_analysis.py`).
- **Explanations.** `ExtendedIsolationForest.explain` gives a per-sensor importance for every
  row. At each split on the row's path, each sensor gets its share of the split value
  `|n_k (x_k - p_k)|`. That share is weighted by how strongly the split isolates the row
  (`1 - size(child) / size(node)`), and the result is averaged over trees.

  **What it is not:** this is a simplified heuristic *inspired by* ExIFFI. It is not a
  reimplementation of ExIFFI. It is compared with Local-DIFFI (implemented from Algorithm 3
  of the DIFFI paper and checked against a step-by-step reference implementation in the
  tests), a z-score and random ranking in the synthetic test above.
- **Streaming.** A Redpanda and Docker Compose demo replays an experiment row by row. It
  uses the same 300 + 100 warm-up, and it logs alarms with their top sensors. See
  [`docs/streaming.md`](docs/streaming.md).

## Limitations

- **High false-alarm rate.** The default EIF flags about half of the normal test rows,
  mostly after the anomaly, when the plant has settled at a new state. Dropping Thermocouple
  lowers this to about a third, at the cost of more missed anomalous rows. Smoothing and
  linear detrending did not help.
- **Per-experiment fitting.** Every experiment gets its own model, fit on its first 300
  rows (5 minutes at 1 Hz). This is not one model deployed across runs.
- **Simplified explanations.** On shifted sensors they add nothing over a z-score. The
  synthetic test covers two kinds of anomaly only (mean shifts and broken pairs), and the
  injected rows are real normal rows, not real faults.
- **Streaming is a demo only.** It replays recorded data with a fixed model and has no
  separate evaluation.

## Reproduce

```bash
make install                          # in a Python 3.12 virtualenv; versions are pinned
make data                             # clones SKAB into data/raw/SKAB (not committed)
make test && make lint
python scripts/skab_benchmark.py      # docs/skab_*.csv, SKAB protocol (about 10 minutes)
python scripts/false_alarm_analysis.py raw no_thermo thermo_rate   # where false alarms fall
python scripts/explanation_benchmark.py   # docs/explanation_benchmark*.csv (about 5 minutes)
python scripts/compare_models.py      # docs/compare_if_vs_eif*.csv, per-experiment protocol
python scripts/run_baseline.py        # docs/baseline_*.csv
python scripts/explain_skab.py        # docs/explanation_* figures and CSVs
python scripts/results_table.py       # the README table, from the CSVs
bash scripts/check_project.sh         # tests, lint, clean tree, README matches CSVs
make stream                           # streaming demo in Docker Compose
```

All runs are seeded. The tests use synthetic data and do not need SKAB.

## Data and attribution

[SKAB, the Skoltech Anomaly Benchmark](https://github.com/waico/SKAB) (Katser and Kozitsin,
2020) records 8 sensors at 1 Hz on a water-circulation test bench. SKAB is licensed under
**GPL-3.0**. This repository **downloads it at build time and does not redistribute it**.
The leaderboard rows above are quoted from its README. Dataset notes are in
[`docs/data.md`](docs/data.md).

## Roadmap

- **Re-baseline after the plant settles.** When the score stays high but stable for a long
  time, refit on recent rows, so that a new steady state stops raising alarms. (Training on
  SKAB's anomaly-free recording would not help: its Thermocouple range does not cover the
  experiments.)
- Group consecutive alarms into events.
- A small dashboard for scores, alarms and sensor importance.

## References

- S. Hariri, M. Carrasco Kind, R. J. Brunner. *Extended Isolation Forest.* IEEE Transactions
  on Knowledge and Data Engineering, doi:10.1109/TKDE.2019.2947676.
  [arXiv:1811.02141](https://arxiv.org/abs/1811.02141)
- A. Arcudi, D. Frizzo, C. Masiero, G. A. Susto. *Enhancing Interpretability and
  Generalizability in Extended Isolation Forests* (introduces ExIFFI).
  [arXiv:2310.05468](https://arxiv.org/abs/2310.05468)
- M. Carletti, M. Terzi, G. A. Susto. *Interpretable Anomaly Detection with DIFFI:
  Depth-based Isolation Forest Feature Importance.*
  [arXiv:2007.11117](https://arxiv.org/abs/2007.11117)
- M. Pinet, J. Cumin, S. Berlemont, D. Vaufreydaz. *Anomalies in Multivariate Time Series
  Benchmarks Are Mostly Univariate.* [arXiv:2606.02670](https://arxiv.org/abs/2606.02670)
- I. D. Katser, V. O. Kozitsin. *Skoltech Anomaly Benchmark (SKAB)*, 2020.
  <https://github.com/waico/SKAB>

## License

MIT for the code in this repository (see [LICENSE](LICENSE)). SKAB keeps its own GPL-3.0
licence.
