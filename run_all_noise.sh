#!/usr/bin/env bash
set -euo pipefail

# Run all experiments in URAG/configs with concurrency, skipping finished ones.
# Finished experiments are detected by checking if their output directory contains
# result files like calibration, test, or evaluate files.
# Usage:
#   ./URAG/run_all.sh [CONFIG_DIR] [MAX_JOBS]
# Defaults:
#   CONFIG_DIR = URAG/configs
#   MAX_JOBS   = ${MAX_JOBS:-4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="${1:-$SCRIPT_DIR/configs_noise}"
MAX_JOBS="${2:-${MAX_JOBS:-1}}"
CLI_PY="$SCRIPT_DIR/cli.py"


get_output_dir() {
  local cfg="$1"
  # Extract output directory from YAML config
  if command -v yq >/dev/null 2>&1; then
    yq eval '.output' "$cfg" 2>/dev/null || echo "results"
  else
    # Fallback: grep for output field in YAML
    grep '^output:' "$cfg" 2>/dev/null | sed 's/^output: *//' | tr -d '"' || echo "results"
  fi
}

is_experiment_done() {
  local cfg="$1"
  local output_dir="$(get_output_dir "$cfg")"
  
  # Check if output directory exists
  if [[ ! -d "$output_dir" ]]; then
    return 1
  fi
  
  # Check only for files containing 'evaluate' in their names
  if find "$output_dir" -type f -name "evaluation*" -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  
  return 1
}


is_already_running() {
  # Check if any python process is running cli.py with this config
  local cfg="$1"
  local cfg_basename="$(basename "$cfg")"
  
  if command -v pgrep >/dev/null 2>&1; then
    pgrep -fal "python.*cli.py.*--config.*$cfg_basename" >/dev/null 2>&1 && return 0 || true
  else
    ps aux | grep -E "python.*cli.py.*--config.*$cfg_basename" | grep -v grep >/dev/null 2>&1 && return 0 || true
  fi
  return 1
}

ensure_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    # Wait for any job to finish
    if wait -n 2>/dev/null; then :; else sleep 1; fi
  done
}

run_one() {
  local cfg="$1"
  local base="$(basename "$cfg" .yaml)"

  # Skip finished experiments (check for result files)
  if is_experiment_done "$cfg"; then
    echo "[SKIP-DONE] $cfg (result files exist)"
    return 0
  fi

  # Skip if already running
  if is_already_running "$cfg"; then
    echo "[SKIP-RUNNING] $cfg (process detected)"
    return 0
  fi

  # Launch experiment
  echo "[LAUNCH] $cfg"
  python "$CLI_PY" --config "$cfg" &
}

main() {
  shopt -s nullglob
  mapfile -t configs < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "*.yaml" | sort)
  if (( ${#configs[@]} == 0 )); then
    echo "No YAML configs found in $CONFIG_DIR"
    exit 0
  fi

  echo "Found ${#configs[@]} configs in $CONFIG_DIR; running up to $MAX_JOBS in parallel."

  for cfg in "${configs[@]}"; do
    ensure_slot
    run_one "$cfg"
  done

  # Wait remaining jobs
  wait
  echo "All dispatched experiments completed or skipped."
}

main "$@"

