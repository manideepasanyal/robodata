#!/usr/bin/env bash
# Build both released datasets from curated raw data (assumes ingest + sim + health already ran).
set -euo pipefail
cd "$(dirname "$0")/.."
RD=${ROBODATA_BIN:-robodata}
$RD build ur5_insertion_mixed --shards 8
$RD curate franka_droid100 --workers 8
$RD build franka_droid100 --shards 6
for d in ur5_insertion_mixed franka_droid100; do $RD card "$d"; $RD version "$d" --note "full build"; done
