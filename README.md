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
- **Tuning does not transfer.** Settings were chosen on the 16 valve1 experiments only.
  On the 18 held-out experiments the default EIF scores F1 0.777 (FAR 47.5 %), and the setting tuned on valve1 F1 0.777 (FAR 52.0 %).
  Removing a linear drift fitted on the training rows made results worse. Every setting
  tried is listed in [`docs/skab_grid_dev.csv`](docs/skab_grid_dev.csv).
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
  not single out the anomaly windows.

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
  (0.95, 0.99 or 1.0) and linear detrending (on or off) were chosen on the valve1
  experiments only. They were then evaluated on the held-out valve2 and other experiments.
- **Explanations.** `ExtendedIsolationForest.explain` gives a per-sensor importance for every
  row. At each split on the row's path, each sensor gets its share of the split value
  `|n_k (x_k - p_k)|`. That share is weighted by how strongly the split isolates the row
  (`1 - size(child) / size(node)`), and the result is averaged over trees.

  **What it is not:** this is a simplified heuristic *inspired by* ExIFFI. It is not a
  reimplementation of ExIFFI or DIFFI, and it has not been validated beyond the checks in
  [`docs/results.md`](docs/results.md).
- **Streaming.** A Redpanda and Docker Compose demo replays an experiment row by row. It
  uses the same 300 + 100 warm-up, and it logs alarms with their top sensors. See
  [`docs/streaming.md`](docs/streaming.md).

## Limitations

- **High false-alarm rate.** About half of the normal test rows are flagged, mostly because
  of slow drift after the short training window. Neither smoothing nor linear detrending
  fixed this.
- **Per-experiment fitting.** Every experiment gets its own model, fit on its first 300
  rows (5 minutes at 1 Hz). This is not one model deployed across runs.
- **Simplified explanations.** See above. The check against "visible change" uses a simple
  mean difference, which misses changes in variance or in correlations.
- **Streaming is a demo only.** It replays recorded data with a fixed model and has no
  separate evaluation.

## Reproduce

```bash
make install                          # in a Python 3.12 virtualenv; versions are pinned
make data                             # clones SKAB into data/raw/SKAB (not committed)
make test && make lint
python scripts/skab_benchmark.py      # docs/skab_*.csv, SKAB protocol (about 3 minutes)
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

- **Reduce false alarms from drift.** Options: refit on recent rows that raised no alarm,
  or train on SKAB's long anomaly-free recording so normal variation is better covered.
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
- I. D. Katser, V. O. Kozitsin. *Skoltech Anomaly Benchmark (SKAB)*, 2020.
  <https://github.com/waico/SKAB>

## License

MIT for the code in this repository (see [LICENSE](LICENSE)). SKAB keeps its own GPL-3.0
licence.
