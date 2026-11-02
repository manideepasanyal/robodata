#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
RD=${ROBODATA_BIN:-robodata}
mkdir -p data/logs data/reports/health
for r in $(grep -v "^#" configs/hub_datasets.txt | awk '{print $1}'); do
  $RD health --robot "$r" --workers 14 --quarantine > "data/logs/health_$r.log" 2>&1
  cp data/reports/health/health_summary.csv "data/reports/health/health_summary_$r.csv"
  echo "$r $(tail -6 data/logs/health_$r.log | tr '\n' ' ')"
done
