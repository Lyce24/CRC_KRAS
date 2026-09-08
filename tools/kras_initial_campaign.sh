#!/usr/bin/env bash
# Initial end-to-end KRAS campaign (2026-08-14, study owner directive):
# freeze the budget-grid HP winner, then P1 → S1 → S2 → B1 → B2 (3 seeds x
# 5-fold OOF CV each, dev report per experiment) → phase 6 EXT-P + gatekeeping
# → phase 7 T1. Every stage is resumable; re-running this script skips
# completed work. Stage markers ("=== STAGE ...") are machine-readable for
# monitoring.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

stage() {  # stage <name> <cmd...>
  local name=$1; shift
  echo "=== STAGE START ${name} $(date '+%H:%M:%S')"
  if "$@"; then
    echo "=== STAGE DONE ${name} $(date '+%H:%M:%S')"
  else
    echo "=== STAGE FAILED ${name} (exit $?) $(date '+%H:%M:%S')"
    exit 1
  fi
}

# Wait for any in-flight tuning process to drain (GPU serialization).
while pgrep -f "phase4_hyperparameter_tuning.py run" > /dev/null; do
  sleep 30
done

stage phase4-report        $PY phase4_hyperparameter_tuning.py report

for exp in p1 s1 s2 b1 b2; do
  stage "train-${exp}"     $PY phase5_experiment_dev_cv.py train "${exp}"
  stage "report-${exp}"    $PY phase5_experiment_dev_cv.py report "${exp}"
done

stage ext-p-score          $PY phase6_external_primary.py score
stage ext-p-report         $PY phase6_external_primary.py report
stage t1-score             $PY phase7_metastatic_transfer.py score
stage t1-report            $PY phase7_metastatic_transfer.py report

echo "=== CAMPAIGN COMPLETE $(date '+%H:%M:%S')"
