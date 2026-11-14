"""Score SKAB rows from a Kafka/Redpanda topic with the EIF and log alarms with their top sensors."""

import argparse
import logging
import os

from kafka import KafkaConsumer

from xad.stream import StreamingDetector, decode

log = logging.getLogger("consumer")


def main() -> None:
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", default=env("BOOTSTRAP", "localhost:9092"))
    parser.add_argument("--topic", default=env("TOPIC", "skab"))
    parser.add_argument("--warmup", type=int, default=int(env("WARMUP", "400")))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("kafka").setLevel(logging.WARNING)  # hide connection chatter

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        group_id="xad-consumer",
        auto_offset_reset="earliest",
    )
    detector = StreamingDetector(warmup=args.warmup)
    log.info("warm-up: the first %d rows are assumed normal", args.warmup)
    counts = {"scored": 0, "alarms": 0, "true_alarms": 0, "anomalous": 0}
    expected_row = 0

    for record in consumer:
        try:
            message = decode(record.value)
            if message.get("end"):
                break
            result = detector.update(message["sensors"])
        except (KeyError, TypeError, ValueError) as err:
            log.warning("skipping bad message at offset %d: %r", record.offset, err)
            continue
        if message.get("row") != expected_row:
            log.warning("row %s arrived, expected %d", message.get("row"), expected_row)
        expected_row += 1
        if result is None:
            if detector.ready:
                log.info("warm-up done, threshold %.3f", detector.threshold)
            continue

        label = message.get("anomaly")
        counts["scored"] += 1
        counts["anomalous"] += label == 1
        if result["alarm"]:
            counts["alarms"] += 1
            counts["true_alarms"] += label == 1
            top = ", ".join(f"{name} {share:.2f}" for name, share in result["top"])
            log.info(
                "ALARM row=%s score=%.3f > %.3f | top: %s | label=%s",
                message.get("row"),
                result["score"],
                result["threshold"],
                top,
                label,
            )

    consumer.close()
    log.info(
        "end of stream: %(scored)d rows scored, %(alarms)d alarms "
        "(%(true_alarms)d on labelled-anomalous rows), %(anomalous)d labelled-anomalous rows",
        counts,
    )


if __name__ == "__main__":
    main()
