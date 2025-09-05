#!/usr/bin/env bash
set -euo pipefail

# Run all experiments in URAG/configs with concurrency, skipping running or finished ones.
# Usage:
#   ./URAG/run_all.sh [CONFIG_DIR] [MAX_JOBS]
# Defaults:
#   CONFIG_DIR = URAG/configs
#   MAX_JOBS   = ${MAX_JOBS:-4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="${1:-$SCRIPT_DIR/configs}"
MAX_JOBS="${2:-${MAX_JOBS:-4}}"
STATUS_DIR="$SCRIPT_DIR/.run_status"
LOG_DIR="$SCRIPT_DIR/logs"
CLI_PY="$SCRIPT_DIR/cli.py"

mkdir -p "$STATUS_DIR" "$LOG_DIR"

safe_name() {
  # Create a filesystem-safe unique name for a path
  local path="$1"
  # Prefer md5sum for stability
  if command -v md5sum >/dev/null 2>&1; then
    echo -n "$path" | md5sum | awk '{print $1}'
  else
    # Fallback: replace slashes
    echo "$path" | sed 's#[/ ]#_#g'
  fi
}

lock_file() {
  local cfg="$1"
  echo "$STATUS_DIR/$(safe_name "$cfg").lock"
}

done_file() {
  local cfg="$1"
  echo "$STATUS_DIR/$(safe_name "$cfg").done"
}

is_pid_running() {
  local pid="$1"
  if [[ -z "$pid" ]]; then return 1; fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

is_already_running_elsewhere() {
  # Best-effort: detect any python process running cli.py with this config
  local cfg="$1"
  if command -v pgrep >/dev/null 2>&1; then
    pgrep -fal "python .*cli.py .*--config .*$(printf '%q' "$cfg")" >/dev/null 2>&1 && return 0 || true
    pgrep -fal "python .*URAG/cli.py .*--config .*$(printf '%q' "$cfg")" >/dev/null 2>&1 && return 0 || true
  else
    ps aux | grep -E "python .*cli.py .*--config .*$(printf '%q' "$cfg")" | grep -v grep >/dev/null 2>&1 && return 0 || true
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
  local lock="$(lock_file "$cfg")"
  local donef="$(done_file "$cfg")"
  local base="$(basename "$cfg" .yaml)"
  local log="$LOG_DIR/run_${base}.log"

  # Skip finished
  if [[ -f "$donef" ]]; then
    echo "[SKIP-DONE] $cfg"
    return 0
  fi

  # Handle lock
  if [[ -f "$lock" ]]; then
    local pid
    pid="$(cat "$lock" 2>/dev/null || true)"
    if is_pid_running "$pid"; then
      echo "[SKIP-RUNNING] $cfg (pid=$pid)"
      return 0
    else
      echo "[STALE-LOCK] $cfg (pid=$pid) -> removing"
      rm -f "$lock"
    fi
  fi

  # Also skip if detected running elsewhere
  if is_already_running_elsewhere "$cfg"; then
    echo "[SKIP-RUNNING-ELSEWHERE] $cfg"
    return 0
  fi

  # Launch wrapper to ensure lock/done management
  echo "[LAUNCH] $cfg"
  bash -c "\
    set -euo pipefail; \
    echo $$ > '$lock'; \
    (
      python '$CLI_PY' --config '$cfg' 2>&1 | tee '$log'
    ); \
    touch '$donef'; \
    rm -f '$lock' \
  " &
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
  echo "All dispatched experiments completed or skipped. Logs: $LOG_DIR"
}

main "$@"
