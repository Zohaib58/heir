#!/usr/bin/env bash

set -u
set -o pipefail

ROOT="${HEIR_ROOT:-/home/zohaib/heir-private}"
RESULTS="$ROOT/benchmark-results"
LOGS="$RESULTS/logs"
RUNNER="$ROOT/tests/CROSS/run_heir_model_benchmark.py"

export CROSS_ROOT="${CROSS_ROOT:-/home/zohaib/CROSS_dev}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOGS"
cd "$ROOT"

run_degree() {
  local label="$1"
  local model="alexnet-tiny-$label"
  local correctness_dir="$RESULTS/alexnet-$label-cpu-correctness"
  local operations_dir="$RESULTS/alexnet-$label-cpu-operations"
  local correctness_log="$LOGS/alexnet-tiny-$label-cpu-correctness.log"
  local operations_log="$LOGS/alexnet-tiny-$label-cpu-operations.log"

  printf '\n[%s] Starting %s CPU correctness\n' "$(date -u +%FT%TZ)" "$model"
  if python3 "$RUNNER" \
      --model "$model" \
      --device cpu \
      --warmups 0 \
      --repeats 1 \
      --output-dir "$correctness_dir" \
      2>&1 | tee "$correctness_log"; then
    printf '[%s] %s correctness passed; starting operation profile\n' \
      "$(date -u +%FT%TZ)" "$model"
    python3 "$RUNNER" \
      --model "$model" \
      --device cpu \
      --profile-only \
      --profile-callsites \
      --output-dir "$operations_dir" \
      2>&1 | tee "$operations_log"
    local profile_status=${PIPESTATUS[0]}
    printf '[%s] %s operation profile exited with status %d\n' \
      "$(date -u +%FT%TZ)" "$model" "$profile_status"
  else
    local correctness_status=${PIPESTATUS[0]}
    printf '[%s] %s correctness failed with status %d; skipping its profile\n' \
      "$(date -u +%FT%TZ)" "$model" "$correctness_status"
  fi
}

run_degree 16k
run_degree 32k

printf '\n[%s] AlexNet CPU queue complete\n' "$(date -u +%FT%TZ)"
