#!/usr/bin/env bash
# Build + independently audit every dataset recipe (hub datasets + the two original releases).
set -uo pipefail
cd "$(dirname "$0")/.."
RD=${ROBODATA_BIN:-robodata}
mkdir -p data/logs data/reports/health
$RD curate ur5_insertion_mixed --workers 14 > data/logs/curate_ur5_insertion_mixed.log 2>&1
$RD curate franka_droid100 --workers 14 > data/logs/curate_franka_droid100.log 2>&1
for r in $(grep -v "^#" configs/hub_datasets.txt | awk '{print $1}') ur5_insertion_mixed franka_droid100; do
  start=$(date +%s)
  if $RD build "$r" --shards 6 > "data/logs/build_$r.log" 2>&1; then b=ok; else b=FAILED; fi
  $RD audit "$r" > "data/logs/audit_$r.log" 2>&1
  echo "$r build=$b $(( $(date +%s) - start ))s audit=$(head -1 data/logs/audit_$r.log)"
done
