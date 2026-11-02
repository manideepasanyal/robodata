#!/usr/bin/env bash
# Scale-out validation on 15 LeRobot hub datasets, from nothing to audited datasets:
# exact pinned downloads -> virtual-robot ingest -> planted duplicates + faults (ground truth)
# -> health + quarantine -> curation/dedup -> build -> independent audit vs. the original data.
set -euo pipefail
cd "$(dirname "$0")/.."
RD=${ROBODATA_BIN:-robodata}
mkdir -p data/logs data/reports
names=$(grep -v "^#" configs/hub_datasets.txt | awk '{print $1}')
for s in $names; do $RD fetch --source "$s"; done          # pinned revision + episode subset from configs/sources
bash scripts/ingest_hub.sh
for s in $names; do $RD dupes --robot "$s" --n 3; done      # 18 planted duplicates per dataset
for s in pusht aloha_static_cups_open libero_10_image; do $RD faults --robot "$s" --episodes 1; done
bash scripts/health_hub.sh
bash scripts/curate_hub.sh
bash scripts/build_audit_all.sh                              # also (re)curates + builds the two original datasets
$RD dedup-eval
