#!/usr/bin/env bash
set -euo pipefail

###############################################
# Create a 10-plan subset of Perseus results  #
###############################################
#
# Usage:
#   ./create_perseus_results10.sh [perseus_results_dir]
#
# Defaults:
#   perseus_results_dir:
#     tests/kareus/llama3.2_3b/cp2_tp4_bs16_seq4096/perseus_results
#   NUM_SAMPLES (env): 10
#
# This script:
#   - enumerates freqs_pipeline_*.py and scheds_pipeline_*.py
#   - samples up to NUM_SAMPLES of each (using stride from largest index)
#   - moves the selected plans into:
#       <perseus_results_dir>/perseus_results10
###############################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="llama3.2_3b"
CONFIG="cp2_tp4_bs8_seq8192"

DEFAULT_PERSEUS_DIR="${SCRIPT_DIR}/${MODEL_NAME}/${CONFIG}/perseus_results"
PERSEUS_DIR="${1:-$DEFAULT_PERSEUS_DIR}"

if [[ ! -d "$PERSEUS_DIR" ]]; then
  echo "ERROR: perseus_results directory not found: $PERSEUS_DIR" >&2
  exit 1
fi

NUM_SAMPLES=${NUM_SAMPLES:-15}

echo "Using perseus results directory: $PERSEUS_DIR"
echo "NUM_SAMPLES=${NUM_SAMPLES}"

###############################################
# Enumerate freqs and scheds plans            #
###############################################

mapfile -t FREQ_FILES < <(ls -1 "${PERSEUS_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort)
# mapfile -t SCHED_FILES < <(ls -1 "${PERSEUS_DIR}"/scheds_pipeline_*.py 2>/dev/null | sort)

if [[ ${#FREQ_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No freqs_pipeline_*.py found under $PERSEUS_DIR" >&2
  exit 1
fi

# if [[ ${#SCHED_FILES[@]} -eq 0 ]]; then
#   echo "ERROR: No scheds_pipeline_*.py found under $PERSEUS_DIR" >&2
#   exit 1
# fi

TOTAL_FREQ_FILES=${#FREQ_FILES[@]}
# TOTAL_SCHED_FILES=${#SCHED_FILES[@]}

echo "Found ${TOTAL_FREQ_FILES} freqs plans"
# echo "Found ${TOTAL_SCHED_FILES} scheds plans"

###############################################
# Sample freqs plans                          #
###############################################

# Old logic (NUM_SAMPLES-based stride), now disabled:
if (( NUM_SAMPLES > 0 && NUM_SAMPLES < TOTAL_FREQ_FILES )); then
  # Compute stride so that we pick at most NUM_SAMPLES plans
  stride=$(( (TOTAL_FREQ_FILES + NUM_SAMPLES - 1) / NUM_SAMPLES ))
  (( stride < 1 )) && stride=1
  echo "Sampling ${NUM_SAMPLES} freqs plans from ${TOTAL_FREQ_FILES} total (computed stride=${stride})"
  declare -a STRIDED_FREQ_FILES=()
  for ((i=TOTAL_FREQ_FILES-1; i>=0; i-=stride)); do
    STRIDED_FREQ_FILES+=("${FREQ_FILES[$i]}")
  done
  FREQ_FILES=("${STRIDED_FREQ_FILES[@]}")
fi

# New logic: fixed stride of 100, then cap to NUM_SAMPLES (default 10).
# FIXED_FREQ_STRIDE=100
# declare -a STRIDED_FREQ_FILES=()
# for ((i=TOTAL_FREQ_FILES-1; i>=0; i-=FIXED_FREQ_STRIDE)); do
#   STRIDED_FREQ_FILES+=("${FREQ_FILES[$i]}")
# done

# if (( NUM_SAMPLES > 0 && ${#STRIDED_FREQ_FILES[@]} > NUM_SAMPLES )); then
#   echo "Sampling ${NUM_SAMPLES} freqs plans from ${TOTAL_FREQ_FILES} total (fixed stride=${FIXED_FREQ_STRIDE})"
#   FREQ_FILES=("${STRIDED_FREQ_FILES[@]:0:NUM_SAMPLES}")
# else
#   echo "Sampling ${#STRIDED_FREQ_FILES[@]} freqs plans from ${TOTAL_FREQ_FILES} total (fixed stride=${FIXED_FREQ_STRIDE})"
#   FREQ_FILES=("${STRIDED_FREQ_FILES[@]}")
# fi

###############################################
# Sample scheds plans                         #
###############################################

# # Old logic (NUM_SAMPLES-based stride), now disabled:
# if (( NUM_SAMPLES > 0 && NUM_SAMPLES < TOTAL_SCHED_FILES )); then
#   # Compute stride so that we pick at most NUM_SAMPLES plans
#   stride=$(( (TOTAL_SCHED_FILES + NUM_SAMPLES - 1) / NUM_SAMPLES ))
#   (( stride < 1 )) && stride=1
#   echo "Sampling ${NUM_SAMPLES} scheds plans from ${TOTAL_SCHED_FILES} total (computed stride=${stride})"
#   declare -a STRIDED_SCHED_FILES=()
#   for ((i=TOTAL_SCHED_FILES-1; i>=0; i-=stride)); do
#     STRIDED_SCHED_FILES+=("${SCHED_FILES[$i]}")
#   done
#   SCHED_FILES=("${STRIDED_SCHED_FILES[@]}")
# fi

# # New logic: fixed stride of 100, then cap to NUM_SAMPLES (default 10).
# FIXED_SCHED_STRIDE=100
# declare -a STRIDED_SCHED_FILES=()
# for ((i=TOTAL_SCHED_FILES-1; i>=0; i-=FIXED_SCHED_STRIDE)); do
#   STRIDED_SCHED_FILES+=("${SCHED_FILES[$i]}")
# done

# if (( NUM_SAMPLES > 0 && ${#STRIDED_SCHED_FILES[@]} > NUM_SAMPLES )); then
#   echo "Sampling ${NUM_SAMPLES} scheds plans from ${TOTAL_SCHED_FILES} total (fixed stride=${FIXED_SCHED_STRIDE})"
#   SCHED_FILES=("${STRIDED_SCHED_FILES[@]:0:NUM_SAMPLES}")
# else
#   echo "Sampling ${#STRIDED_SCHED_FILES[@]} scheds plans from ${TOTAL_SCHED_FILES} total (fixed stride=${FIXED_SCHED_STRIDE})"
#   SCHED_FILES=("${STRIDED_SCHED_FILES[@]}")
# fi

###############################################
# Move selected plans into perseus_results10  #
###############################################

TARGET_DIR="${PERSEUS_DIR}/../frontier"
mkdir -p "${TARGET_DIR}"

echo "Moving ${#FREQ_FILES[@]} freqs plans to ${TARGET_DIR}"
for f in "${FREQ_FILES[@]}"; do
  mv "$f" "${TARGET_DIR}/"
done

# echo "Moving ${#SCHED_FILES[@]} scheds plans to ${TARGET_DIR}"
# for s in "${SCHED_FILES[@]}"; do
#   mv "$s" "${TARGET_DIR}/"
# done

echo "Done. Subset written to ${TARGET_DIR}"


