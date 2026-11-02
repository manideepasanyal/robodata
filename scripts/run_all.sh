#!/usr/bin/env bash
# Full pipeline from nothing: downloads -> raw -> health -> curated -> datasets -> versions -> training smoke test.
set -euo pipefail
cd "$(dirname "$0")/.."
RD=${ROBODATA_BIN:-robodata}
$RD fetch --source catheter
$RD fetch --source droid100
$RD ingest catheter
$RD ingest droid100
$RD sim generate --episodes-per-scene 10
$RD sim convert
$RD sim calibrate
$RD faults --robot ur5_catheter --episodes 2      # fault-injected copies validate the health checks
$RD faults --robot ur5_sim --episodes 1
$RD faults --robot franka_droid --episodes 1
$RD health --quarantine
$RD curate ur5_insertion_mixed
bash scripts/build_all.sh
$RD inspect ur5_insertion_mixed --video
$RD bench video && $RD bench formats --episodes 25
$RD train-smoke ur5_insertion_mixed --steps 300 --sim-weight 0.3
$RD derive ur5_insertion_mixed --name ur5_insertion_mixed_rgb --drop-depth
