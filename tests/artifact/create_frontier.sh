#!/usr/bin/env bash
set -euo pipefail

###############################################
# Create a frontier subset of optimisation    #
# results for kareus / nanobatch / perseus.   #
###############################################
#
# Usage:
#   ./create_frontier.sh <method>
#
# Where <method> is one of:
#   - kareus       (source: kareus/lowtime,            target: tests/kareus/schedules)
#   - nanobatch    (source: nanobatch_perseus/lowtime, target: tests/nanobatch_perseus/schedules)
#   - perseus      (source: perseus/lowtime,           target: tests/perseus/schedules)
#
# For "kareus", both freqs_pipeline_*.py and scheds_pipeline_*.py are
# processed. For "nanobatch" and "perseus", only freqs_pipeline_*.py.
#
# Sampling logic:
#   1. Sort plans by index ascending and pick 15 strided samples
#      (largest-index-first stride identical to create_perseus_results10.sh,
#      then re-sorted ascending).
#   2. From the 15 samples, drop the 1st, 3rd, 5th, 7th, 9th
#      (0-indexed positions 0, 2, 4, 6, 8), leaving 10 plans
#      at sample positions 1, 3, 5, 7, 9, 10, 11, 12, 13, 14.
#
# Optional env vars:
#   MODEL_NAME   default: megatron_llama3.2_3b
#   CONFIG_TAG   default: cp1_tp8_mbs8_seq4096
#   NUM_SAMPLES  default: 15
###############################################

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <method: kareus|nanobatch|perseus>" >&2
  exit 1
fi

METHOD="$1"

case "${METHOD}" in
  kareus)
    SOURCE_SUBDIR="kareus"
    TARGET_BASE_SUBDIR="kareus"
    PROCESS_SCHEDS=true
    ;;
  nanobatch)
    SOURCE_SUBDIR="nanobatch_perseus"
    TARGET_BASE_SUBDIR="nanobatch_perseus"
    PROCESS_SCHEDS=false
    ;;
  perseus)
    SOURCE_SUBDIR="perseus"
    TARGET_BASE_SUBDIR="perseus"
    PROCESS_SCHEDS=false
    ;;
  *)
    echo "ERROR: unknown method '${METHOD}'. Use one of: kareus, nanobatch, perseus." >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-megatron_llama3.2_3b}"
CONFIG_TAG="${CONFIG_TAG:-cp1_tp8_mbs8_seq4096}"
NUM_SAMPLES="${NUM_SAMPLES:-15}"

SOURCE_DIR="${SCRIPT_DIR}/nemo_experiments/${MODEL_NAME}/${CONFIG_TAG}/${SOURCE_SUBDIR}/lowtime"
TARGET_DIR="${TESTS_DIR}/${TARGET_BASE_SUBDIR}/schedules/${MODEL_NAME}/${CONFIG_TAG}/frontier"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "ERROR: source lowtime directory not found: ${SOURCE_DIR}" >&2
  exit 1
fi

echo "Method:      ${METHOD}"
echo "Source dir:  ${SOURCE_DIR}"
echo "Target dir:  ${TARGET_DIR}"
echo "NUM_SAMPLES: ${NUM_SAMPLES}"
echo "Process scheds: ${PROCESS_SCHEDS}"

mkdir -p "${TARGET_DIR}"

###############################################
# Helper: stride-sample then drop odd-indexed #
###############################################
# Args: <pattern_glob_basename>  (e.g. freqs_pipeline_*.py)
# Reads files from SOURCE_DIR matching the pattern, applies the
# stride described above, drops 0-indexed positions {0,2,4,6,8},
# and copies the survivors into TARGET_DIR.
sample_and_copy() {
  local pattern="$1"
  local label="$2"

  mapfile -t ALL_FILES < <(ls -1 "${SOURCE_DIR}"/${pattern} 2>/dev/null | sort)
  local total=${#ALL_FILES[@]}
  if (( total == 0 )); then
    echo "ERROR: No ${pattern} found under ${SOURCE_DIR}" >&2
    exit 1
  fi
  echo "Found ${total} ${label} plans"

  declare -a SAMPLED=()
  if (( NUM_SAMPLES > 0 && NUM_SAMPLES < total )); then
    local stride=$(( (total + NUM_SAMPLES - 1) / NUM_SAMPLES ))
    (( stride < 1 )) && stride=1
    echo "  Sampling ${NUM_SAMPLES} ${label} plans from ${total} (stride=${stride})"
    for ((i=total-1; i>=0; i-=stride)); do
      SAMPLED+=("${ALL_FILES[$i]}")
    done
    # Re-sort ascending for stable, predictable index removal.
    mapfile -t SAMPLED < <(printf '%s\n' "${SAMPLED[@]}" | sort)
  else
    echo "  Using all ${total} ${label} plans (no stride)"
    SAMPLED=("${ALL_FILES[@]}")
  fi

  echo "  Sampled ${#SAMPLED[@]} ${label} plans before dropping"

  # Drop 0-indexed positions {0, 2, 4, 6, 8}.
  local drop_set=" 0 2 4 6 8 "
  declare -a KEPT=()
  for ((i=0; i<${#SAMPLED[@]}; i++)); do
    if [[ "${drop_set}" != *" ${i} "* ]]; then
      KEPT+=("${SAMPLED[$i]}")
    fi
  done

  echo "  Dropping positions {0,2,4,6,8}: keeping ${#KEPT[@]} ${label} plans"
  for f in "${KEPT[@]}"; do
    cp "$f" "${TARGET_DIR}/"
  done
}

sample_and_copy "freqs_pipeline_*.py" "freqs"

if [[ "${PROCESS_SCHEDS}" == "true" ]]; then
  sample_and_copy "scheds_pipeline_*.py" "scheds"
fi

echo "Done. Frontier written to: ${TARGET_DIR}"
