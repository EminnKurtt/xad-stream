# Streaming demo (v0.2 preview)

This demo replays one recorded SKAB experiment through a Kafka-compatible broker, Redpanda.
A consumer scores each row as it arrives and logs alarms with the sensors that drive them.
It shows the detector in a streaming setting. It is not a new evaluation: the metrics are
the offline ones in [`results.md`](results.md).

```text
SKAB CSV --> producer --(topic "skab", one JSON message per row)--> Redpanda --> consumer --> alarm log
```

## Run

```bash
make data                                  # SKAB into data/raw/SKAB (mounted read-only)
make stream                                # docker compose up --build
RATE=1 EXPERIMENT=valve2/0 make stream     # real time (1 Hz), another experiment
make stream-down                           # docker compose down (also deletes the topic)
```

- `RATE` is rows per second. The default is 10, so `valve1/0` (1,147 rows) takes about
  2 minutes. Set `RATE=1` to replay in real time, or `RATE=0` to replay as fast as possible.
- The consumer stops at the end-of-stream message and prints a summary. Redpanda keeps
  running until `make stream-down`. Run `make stream-down` between runs, so the next
  consumer does not re-read old messages.

## What the consumer does

[`xad.stream.StreamingDetector`](../src/xad/stream.py) holds the logic. Kafka is kept out of
that module, so it is tested without a broker in [`tests/test_stream.py`](../tests/test_stream.py).

1. **Warm-up.** The first 400 rows are assumed to be normal operation. The scaler and the
   EIF are fit on the first 300 of them. The alarm threshold is the 0.95 quantile of the
   scores on the last 100. This is the same protocol as the offline comparison, and a test
   checks that the streaming scores equal the batch scores.
2. **Scoring.** Every later row gets an EIF score. A row scoring above the threshold is
   logged as an alarm, with its top 3 sensors by share of `explain` importance (the
   simplified, ExIFFI-inspired heuristic). The label is logged next to the alarm for
   comparison only. The detector never uses labels.
3. **Bad messages.** Malformed JSON, missing sensors and non-finite values are logged and
   skipped. They never change the detector's state.

Alarm log format:

```text
consumer ALARM row=<row> score=<score> > <threshold> | top: <sensor> <share>, <sensor> <share>, <sensor> <share> | label=<0|1>
```

## Limitations

- **The warm-up assumption is labels-free but strong.** It holds for 33 of the 34
  experiments. In `other/2` the first anomaly comes at row 104, inside the warm-up.
- **No refitting.** The model is fixed after warm-up, so slow drift produces false alarms,
  as described in `results.md`.
- **One message per alarm row.** There is no debouncing or alarm grouping yet.
- **Single partition, local broker.** The setup is minimal, with no security or
  persistence: a demo, not a deployment.
