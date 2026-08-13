#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIC_BAYES_ROOT="${HIC_BAYES_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
source "$HIC_BAYES_ROOT/config/env.sh"
PYTHON="${PYTHON:-$HIC_BAYES_ROOT/.venv/bin/python}"
PARENT_SCHEDD="${PARENT_SCHEDD:-$(hostname -s 2>/dev/null || hostname)}"

ANALYSIS_DIR="${SCRIPT_DIR#$HIC_BAYES_ROOT/}"
SYSTEM_NAME="$(basename "$(dirname "$ANALYSIS_DIR")")"
STUDY_NAME="$(basename "$ANALYSIS_DIR")"
RUN_ROOT="$HIC_BAYES_RESULTS/$SYSTEM_NAME/$STUDY_NAME"
RESULTS_DIR="$RUN_ROOT/chains"
LOG_DIR="$RUN_ROOT/logs"
PLOTS_DIR="$RUN_ROOT/plots"
PLOT_MONITOR_SUBMIT_FILE="$SCRIPT_DIR/scripts/condor_uncertainty_correlation_plot_monitor.sub"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) RUN_ID="$2"; shift 2 ;;
        -h|--help) echo "Usage: $(basename "$0") [--run-id run_001]"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${RUN_ID:-}" ]]; then
    latest_run="$(find "$RESULTS_DIR" -maxdepth 1 -type d -name 'run_[0-9][0-9][0-9]' -exec basename {} \; | sort | tail -n 1)"
    [[ -n "$latest_run" ]] || { echo "ERROR: no run_XXX directories found under $RESULTS_DIR" >&2; exit 2; }
    RUN_ID="$latest_run"
fi

[[ -d "$RESULTS_DIR/$RUN_ID" ]] || { echo "ERROR: results run not found: $RESULTS_DIR/$RUN_ID" >&2; exit 2; }
mkdir -p "$LOG_DIR/$RUN_ID" "$PLOTS_DIR/$RUN_ID"

echo "Submitting plot-only job for $SYSTEM_NAME/$STUDY_NAME $RUN_ID..."
condor_submit   -append "HIC_BAYES_ROOT=$HIC_BAYES_ROOT"   -append "PYTHON=$PYTHON"   -append "ANALYSIS_DIR=$ANALYSIS_DIR"   -append "HIC_BAYES_RESULTS=$HIC_BAYES_RESULTS"   -append "RUN_ROOT=$RUN_ROOT"   -append "RESULTS_DIR=$RESULTS_DIR"   -append "LOG_DIR=$LOG_DIR"   -append "PLOTS_DIR=$PLOTS_DIR"   -append "RUN_ID=$RUN_ID"   -append "PARENT_CLUSTER=plot-only"   -append "EXPECTED_JOBS=0"   -append "PARENT_SCHEDD=$PARENT_SCHEDD"   "$PLOT_MONITOR_SUBMIT_FILE"
