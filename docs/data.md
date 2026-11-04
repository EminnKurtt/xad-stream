# Data: SKAB (Skoltech Anomaly Benchmark)

Source: https://github.com/waico/SKAB (downloaded with `make data` into `data/raw/SKAB`; not committed).
License: see the upstream repository (to be confirmed and cited in the README).

## What it is
Multivariate sensor time series from a water-circulation test bench, recorded at 1 Hz.
Each labelled experiment has 8 sensor channels plus two labels:

| Column | Meaning |
|---|---|
| Accelerometer1RMS, Accelerometer2RMS | Vibration (RMS) |
| Current, Voltage | Motor electrical signals |
| Pressure, Volume Flow RateRMS | Hydraulics |
| Temperature, Thermocouple | Temperatures |
| anomaly | 1 if the point is anomalous |
| changepoint | 1 at the start of an anomaly segment |

## What we loaded
- 34 labelled experiments: valve1 (16), valve2 (4), other (14)
- 37,401 rows in total, no missing values
- Anomaly rate 34.9 %, changepoint rate 0.34 %
- The `anomaly-free` folder (normal operation only) is kept for later unsupervised training

## Observations (from `docs/skab_valve1_0.png` and the per-experiment table)
- Vibration, current, voltage and pressure barely change inside the anomaly window of valve1/0.
- The anomaly shows in Temperature (a drop of about 4-5 degrees), Thermocouple (a spike)
  and Volume Flow RateRMS (more readings at 31). A multivariate model is needed.
- Pressure and Volume Flow RateRMS are quantised (few distinct values).
- Every experiment starts with a normal stretch: the first anomalous row is between 495 and 796
  in 33 of 34 experiments; other/2 is the exception (first anomalous row 104).
- Anomaly rate per experiment is close to 35 % in most cases (range 25-49 %).
