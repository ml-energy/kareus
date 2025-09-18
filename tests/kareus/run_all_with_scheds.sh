#!/usr/bin/env bash
set -euo pipefail

# Launch PFO server for each freqs_pipeline_*.py, update YAML to point to the
# corresponding scheds_pipeline_*.py, and run Kareus GPT pretraining.
#
# Usage:
#   ./run_all_with_scheds.sh [results_dir] [yaml_file] [port]
# Defaults:
#   results_dir: tests/kareus/nemo_experiments/megatron_llama_3_2_1b/kareus_perseus_results
#   yaml_file  : tests/kareus/conf/megatron_llama_3_2_1b_config_half.yaml
#   port       : 7787
# Env overrides:
#   SAMPLE_STRIDE   : stride when picking freqs (default 1 = use all)
#   SLEEP_BEFORE_TRAIN : seconds to wait for server readiness (default 3)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

RESULTS_DIR=${1:-$SCRIPT_DIR/nemo_experiments/megatron_llama_3_2_1b/kareus_perseus_results}
YAML_FILE=${2:-$SCRIPT_DIR/conf/megatron_llama_3_2_1b_config_half.yaml}
PORT=${3:-7787}

LOG_DIR=$SCRIPT_DIR/logs_pfo_runs
mkdir -p "$LOG_DIR"

# Discover frequency plan files
mapfile -t FREQ_FILES < <(ls -1 "$RESULTS_DIR"/freqs_pipeline_*.py 2>/dev/null | sort)
if [[ ${#FREQ_FILES[@]} -eq 0 ]]; then
  echo "No freqs_pipeline_*.py found under $RESULTS_DIR" >&2
  exit 1
fi

TOTAL_FILES=${#FREQ_FILES[@]}
echo "Found ${TOTAL_FILES} frequency plans under $RESULTS_DIR"

# Optional sampling by stride from largest to smallest; include the smallest
SAMPLE_STRIDE=${SAMPLE_STRIDE:-20}
if (( SAMPLE_STRIDE > 1 )); then
  echo "Sampling from largest, every ${SAMPLE_STRIDE}th plan (total=${TOTAL_FILES})"
  declare -a STRIDED_FILES=()
  for ((i=TOTAL_FILES-1; i>=0; i-=SAMPLE_STRIDE)); do
    STRIDED_FILES+=("${FREQ_FILES[$i]}")
  done
  if [[ ${#STRIDED_FILES[@]} -eq 0 || "${STRIDED_FILES[-1]}" != "${FREQ_FILES[0]}" ]]; then
    STRIDED_FILES+=("${FREQ_FILES[0]}")
  fi
  FREQ_FILES=("${STRIDED_FILES[@]}")
fi

# Optionally start from a specific plan (resume). Default to freqs_pipeline_02442.py.
# Override with START_AT env var (set to empty to disable).
START_AT=${START_AT:-freqs_pipeline_02442.py}
if [[ -n "$START_AT" ]]; then
  start_idx=-1
  for i in "${!FREQ_FILES[@]}"; do
    if [[ "$(basename "${FREQ_FILES[$i]}")" == "$START_AT" ]]; then
      start_idx=$i; break
    fi
  done
  if (( start_idx >= 0 )); then
    echo "Resuming from $START_AT (index $start_idx of ${#FREQ_FILES[@]})"
    FREQ_FILES=("${FREQ_FILES[@]:$start_idx}")
  else
    echo "START_AT '$START_AT' not found in selected plans; running full selection" >&2
  fi
fi

echo "Running ${#FREQ_FILES[@]} selected plans"

# Backup YAML and restore on exit
BACKUP_YAML="${YAML_FILE}.bak"
cp "$YAML_FILE" "$BACKUP_YAML"
restore_yaml() { mv -f "$BACKUP_YAML" "$YAML_FILE" || true; }
trap restore_yaml EXIT

SLEEP_BEFORE_TRAIN=${SLEEP_BEFORE_TRAIN:-3}

for f in "${FREQ_FILES[@]}"; do
  base=$(basename "$f")
  iter_id=${base#freqs_pipeline_}
  iter_id=${iter_id%.py}
  run_id=${base%.py}
  ts=$(date +%Y%m%d_%H%M%S)

  sched_file="$RESULTS_DIR/scheds_pipeline_${iter_id}.py"
  if [[ ! -f "$sched_file" ]]; then
    echo "[skip] No matching scheds for $base at $sched_file" >&2
    continue
  fi

  echo "=== Running plan: $base with scheds: $(basename "$sched_file") ==="

  # Update YAML solution_path (under kareus_scheduler_kwargs) to sched_file path (relative to tests/kareus)
  # Build path relative to SCRIPT_DIR for consistency with existing YAML
  rel_sched_path=$(python -c 'import os,sys; print(os.path.relpath(os.path.abspath(sys.argv[2]), start=os.path.abspath(sys.argv[1])))' \
    "$SCRIPT_DIR" "$sched_file")

  # Make YAML writable and set solution_path under kareus_scheduler_kwargs with proper indentation
  chmod u+w "$YAML_FILE" || true
  python - "$YAML_FILE" "$rel_sched_path" <<'PY'
import io, os, sys, re
yaml_path = sys.argv[1]
value = sys.argv[2]
with open(yaml_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find kareus_scheduler_kwargs line
idx = None
indent = ''
for i, ln in enumerate(lines):
    m = re.match(r'^(\s*)kareus_scheduler_kwargs:\s*$' , ln)
    if m:
        idx = i
        indent = m.group(1)
        break

if idx is None:
    # Fallback: just append with two-space indent at end
    with open(yaml_path, 'a', encoding='utf-8') as f:
        f.write(f"kareus_scheduler_kwargs:\n  solution_path: {value}\n")
    sys.exit(0)

child_indent = indent + '  '

# Determine the end of kareus_scheduler_kwargs block
block_start = idx + 1
block_end = len(lines)
for j in range(block_start, len(lines)):
    ln = lines[j]
    if ln.strip() == '':
        continue
    if not ln.startswith(child_indent):
        block_end = j
        break

# Search for existing solution_path within the block
replace_j = None
for j in range(block_start, block_end):
    if re.match(r'^\s*solution_path:\s*.*$', lines[j]):
        replace_j = j
        break

new_line = f"{child_indent}solution_path: {value}\n"
if replace_j is not None:
    lines[replace_j] = new_line
else:
    lines.insert(block_start, new_line)
    block_end += 1  # account for insertion

# Remove any other solution_path lines outside the kareus_scheduler_kwargs block
to_delete = []
for i, ln in enumerate(lines):
    if re.match(r'^\s*solution_path:\s*.*$', ln):
        if not (block_start <= i < block_end):
            to_delete.append(i)
for k in reversed(to_delete):
    del lines[k]

with open(yaml_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
PY

  # Start server in background for this freqs plan
  export ZEUS_PFO_SCHEDULER=PointSolution3D
  export ZEUS_PFO_SCHEDULER_ARGS="{\"solution_path\": \"$f\"}"

  server_log="$LOG_DIR/${run_id}_server_${ts}.log"
  echo "Starting PFO server for $base on ${PORT} (log: $server_log)"
  uvicorn zeus.optimizer.pipeline_frequency.server.router:app --port "$PORT" > "$server_log" 2>&1 &
  server_pid=$!

  # Wait for server to become ready
  sleep "$SLEEP_BEFORE_TRAIN"

  train_log="$LOG_DIR/${run_id}_train_${ts}.log"
  echo "Starting training (log: $train_log)"
  if ! python "$SCRIPT_DIR/kareus_gpt_pretraining.py" > "$train_log" 2>&1; then
    echo "Training failed for $base. See $train_log" >&2
  fi

  echo "Stopping server PID $server_pid"
  kill $server_pid >/dev/null 2>&1 || true
  wait $server_pid 2>/dev/null || true

  echo "Completed plan: $base"
  echo
done

echo "All runs completed. Logs in $LOG_DIR"


