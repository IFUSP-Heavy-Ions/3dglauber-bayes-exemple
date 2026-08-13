#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIC_BAYES_ROOT="${HIC_BAYES_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
source "$HIC_BAYES_ROOT/config/env.sh"
PYTHON="${PYTHON:-$HIC_BAYES_ROOT/.venv/bin/python}"

ANALYSIS_DIR="${SCRIPT_DIR#$HIC_BAYES_ROOT/}"
SYSTEM_NAME="$(basename "$(dirname "$ANALYSIS_DIR")")"
STUDY_NAME="$(basename "$ANALYSIS_DIR")"
RUN_ROOT="$HIC_BAYES_RESULTS/$SYSTEM_NAME/$STUDY_NAME"
RESULTS_DIR="$RUN_ROOT/results"
LOG_DIR="$RUN_ROOT/logs"
PLOTS_DIR="$RUN_ROOT/plots"
SUBMIT_FILE="$SCRIPT_DIR/scripts/condor_emulator.sub"

RUN_DESCRIPTION=""
SELECTED_JOBS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --description|-d) RUN_DESCRIPTION="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --only) SELECTED_JOBS+=("$2"); shift 2 ;;
        -h|--help) echo "Usage: $(basename "$0") --description TEXT [--run-id run_001] [--only JOB]"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$RUN_DESCRIPTION" ]] || { echo "ERROR: --description is required" >&2; exit 2; }

if [[ -z "${RUN_ID:-}" ]]; then
    idx=1
    while [[ -d "$RESULTS_DIR/$(printf 'run_%03d' "$idx")" ]]; do idx=$((idx + 1)); done
    RUN_ID="$(printf 'run_%03d' "$idx")"
fi
mkdir -p "$RESULTS_DIR/$RUN_ID" "$LOG_DIR/$RUN_ID" "$PLOTS_DIR/$RUN_ID"

TEMP_SUBMIT_FILE=""
cleanup() { [[ -z "$TEMP_SUBMIT_FILE" ]] || rm -f "$TEMP_SUBMIT_FILE"; }
trap cleanup EXIT

if [[ ${#SELECTED_JOBS[@]} -gt 0 ]]; then
    TEMP_SUBMIT_FILE="$(mktemp)"
    awk 'BEGIN{copy=1} /^queue JOB,PIPELINE,TASK from /{print; print "("; copy=0; next} copy{print}' "$SUBMIT_FILE" > "$TEMP_SUBMIT_FILE"
    for job in "${SELECTED_JOBS[@]}"; do
        match="$(awk -v j="$job" '$1 == j {print}' "$SUBMIT_FILE")"
        [[ -n "$match" ]] || { echo "ERROR: unknown job: $job" >&2; exit 2; }
        echo "$match" >> "$TEMP_SUBMIT_FILE"
    done
    echo ")" >> "$TEMP_SUBMIT_FILE"
    SUBMIT_FILE="$TEMP_SUBMIT_FILE"
fi

condor_submit   -append "HIC_BAYES_ROOT=$HIC_BAYES_ROOT"   -append "PYTHON=$PYTHON"   -append "ANALYSIS_DIR=$ANALYSIS_DIR"   -append "HIC_BAYES_RESULTS=$HIC_BAYES_RESULTS"   -append "RUN_ROOT=$RUN_ROOT"   -append "RESULTS_DIR=$RESULTS_DIR"   -append "LOG_DIR=$LOG_DIR"   -append "PLOTS_DIR=$PLOTS_DIR"   -append "RUN_ID=$RUN_ID"   "$SUBMIT_FILE"
