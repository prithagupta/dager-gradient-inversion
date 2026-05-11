#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common_themis_env.sh"

RUN_SCRIPT="${RUN_SCRIPT:-batch_ablation.sh}"
SCRIPT_ARGS="${SCRIPT_ARGS:-}"
SCRIPT_PATH="$SCRIPT_DIR/$RUN_SCRIPT"
START_TIME="$(date '+%Y-%m-%d %H:%M:%S %Z')"
START_EPOCH="$(date '+%s')"

echo "[THEMIS] Start time: $START_TIME"
echo "[THEMIS] Host: $(hostname)"
echo "[THEMIS] PWD: $(pwd)"
echo "[THEMIS] REPO_ROOT: $REPO_ROOT"
echo "[THEMIS] RUN_SCRIPT: $RUN_SCRIPT"
echo "[THEMIS] SCRIPT_ARGS: $SCRIPT_ARGS"

cd "$REPO_ROOT"

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "[ERROR] Script not found: $SCRIPT_PATH" >&2
  exit 1
fi

if [ -n "$SCRIPT_ARGS" ]; then
  bash "$SCRIPT_PATH" $SCRIPT_ARGS
else
  bash "$SCRIPT_PATH"
fi

END_TIME="$(date '+%Y-%m-%d %H:%M:%S %Z')"
END_EPOCH="$(date '+%s')"
echo "[THEMIS] End time: $END_TIME"
echo "[THEMIS] Elapsed seconds: $((END_EPOCH - START_EPOCH))"
