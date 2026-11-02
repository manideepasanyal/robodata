#!/usr/bin/env bash
# Replay every onboarded LeRobot hub dataset through the virtual robot into data/raw.
set -uo pipefail
cd "$(dirname "$0")/.."
RD=${ROBODATA_BIN:-robodata}
mkdir -p data/logs data/reports/health
for s in $(grep -v "^#" configs/hub_datasets.txt | awk '{print $1}'); do
  start=$(date +%s)
  if $RD ingest "$s" --workers 12 > "data/logs/ingest_$s.log" 2>&1; then status=ok; else status=FAILED; fi
  echo "$s $status $(( $(date +%s) - start ))s $(grep -c '^\[ingest' data/logs/ingest_$s.log) eps"
done
