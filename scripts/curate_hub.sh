#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
RD=${ROBODATA_BIN:-robodata}
mkdir -p data/logs data/reports/health
for r in $(grep -v "^#" configs/hub_datasets.txt | awk '{print $1}'); do
  $RD curate "$r" --workers 14 > "data/logs/curate_$r.log" 2>&1 && echo "$r ok" || echo "$r FAILED"
done
