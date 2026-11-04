#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
if [ -d data/raw/SKAB ]; then
  echo "SKAB already present in data/raw/SKAB"
else
  git clone --depth 1 https://github.com/waico/SKAB.git data/raw/SKAB
fi
