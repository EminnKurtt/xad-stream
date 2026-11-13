FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[stream]"
COPY scripts/stream_producer.py scripts/stream_consumer.py ./scripts/

RUN useradd --create-home app
USER app
