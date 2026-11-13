"""Replay one SKAB experiment to a Kafka/Redpanda topic, one JSON message per row."""

import argparse
import logging
import os
import time
from pathlib import Path

from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError

from xad.data import read_experiment
from xad.stream import END_OF_STREAM, encode, experiment_messages

log = logging.getLogger("producer")


def ensure_topic(bootstrap: str, topic: str) -> None:
    admin = KafkaAdminClient(bootstrap_servers=bootstrap)
    try:
        admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])
        log.info("created topic %s", topic)
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def main() -> None:
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=env("EXPERIMENT", "valve1/0"))
    parser.add_argument("--data-root", default=env("DATA_ROOT", "data/raw/SKAB"))
    parser.add_argument(
        "--rate",
        type=float,
        default=float(env("RATE", "1")),
        help="rows per second (SKAB is recorded at 1 Hz); 0 = as fast as possible",
    )
    parser.add_argument("--bootstrap", default=env("BOOTSTRAP", "localhost:9092"))
    parser.add_argument("--topic", default=env("TOPIC", "skab"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    path = Path(args.data_root) / "data" / f"{args.experiment}.csv"
    if not path.is_file():
        raise SystemExit(f"{path} not found: run `make data` first")
    df = read_experiment(path)

    ensure_topic(args.bootstrap, args.topic)
    producer = KafkaProducer(bootstrap_servers=args.bootstrap, value_serializer=encode)
    log.info("replaying %s (%d rows) at %s rows/s", args.experiment, len(df), args.rate)
    start = time.monotonic()
    for message in experiment_messages(df):
        if args.rate > 0:
            time.sleep(max(0.0, start + message["row"] / args.rate - time.monotonic()))
        producer.send(args.topic, message)
    producer.send(args.topic, END_OF_STREAM)
    producer.flush()
    producer.close()
    log.info("done")


if __name__ == "__main__":
    main()
