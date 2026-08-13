#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIC_BAYES_ROOT="${HIC_BAYES_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python)}"

ANALYSIS_DIR="config/AtlasPbPb2760/mcmc_params"
RESULTS_DIR="$HIC_BAYES_ROOT/$ANALYSIS_DIR/results"
LOG_DIR="$HIC_BAYES_ROOT/$ANALYSIS_DIR/logs"
SUBMIT_FILE="$HIC_BAYES_ROOT/$ANALYSIS_DIR/condor_mcmc_params.sub"
SELECTED_JOBS=()
TEMP_SUBMIT_FILE=""

cleanup() {
    if [[ -n "$TEMP_SUBMIT_FILE" && -f "$TEMP_SUBMIT_FILE" ]]; then
        rm -f "$TEMP_SUBMIT_FILE"
    fi
}
trap cleanup EXIT

usage() {
    cat <<EOF
Usage: $(basename "$0") [--run-id run_001] [--only scan01_minimal]

Options:
  --run-id ID   Submit into an explicit run directory instead of auto-numbering.
  --only JOB    Submit only one scan. May be repeated.
                Jobs: scan01_minimal scan02_short scan03_warmup scan04_medium
                      scan05_long scan06_production_light scan07_production
                      scan08_precision
  -h, --help    Show this help.

You can also set RUN_ID in the environment:
  RUN_ID=run_001 $(basename "$0")
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)
            [[ $# -ge 2 ]] || { echo "ERROR: --run-id requires a value" >&2; exit 2; }
            RUN_ID="$2"
            shift 2
            ;;
        --only)
            [[ $# -ge 2 ]] || { echo "ERROR: --only requires a value" >&2; exit 2; }
            SELECTED_JOBS+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

task_file_for_job() {
    case "$1" in
        scan01_minimal) echo "mcmc_params_PbPb2760_Grad_scan01_minimal.yaml" ;;
        scan02_short) echo "mcmc_params_PbPb2760_Grad_scan02_short.yaml" ;;
        scan03_warmup) echo "mcmc_params_PbPb2760_Grad_scan03_warmup.yaml" ;;
        scan04_medium) echo "mcmc_params_PbPb2760_Grad_scan04_medium.yaml" ;;
        scan05_long) echo "mcmc_params_PbPb2760_Grad_scan05_long.yaml" ;;
        scan06_production_light) echo "mcmc_params_PbPb2760_Grad_scan06_production_light.yaml" ;;
        scan07_production) echo "mcmc_params_PbPb2760_Grad_scan07_production.yaml" ;;
        scan08_precision) echo "mcmc_params_PbPb2760_Grad_scan08_precision.yaml" ;;
        *) return 1 ;;
    esac
}

write_selected_submit_file() {
    TEMP_SUBMIT_FILE="$(mktemp "${TMPDIR:-/tmp}/hic_bayes_mcmc_params.XXXXXX.sub")"

    cat > "$TEMP_SUBMIT_FILE" <<'EOF'
universe     = vanilla
getenv       = True
initialdir   = $(HIC_BAYES_ROOT)

ANALYSIS_DIR = config/AtlasPbPb2760/mcmc_params
TASK_DIR     = $(ANALYSIS_DIR)/tasks
LOG_DIR      = $(ANALYSIS_DIR)/logs

executable   = $(PYTHON)
request_cpus = 6
request_memory = 10000
requirements = !regexp("qgp2", TARGET.Machine) && !regexp("usphydro", TARGET.Machine)
environment  = "HIC_BAYES_RUN_ID=$(RUN_ID)"
+WantFlocking = True

EOF

    for job in "${SELECTED_JOBS[@]}"; do
        task_file="$(task_file_for_job "$job")"
        cat >> "$TEMP_SUBMIT_FILE" <<EOF
arguments    = -m pipelines.run_universality_analysis \$(TASK_DIR)/$task_file
output       = \$(LOG_DIR)/\$(RUN_ID)/condor_\$(ClusterId).\$(ProcId).$job.out
error        = \$(LOG_DIR)/\$(RUN_ID)/condor_\$(ClusterId).\$(ProcId).$job.err
log          = \$(LOG_DIR)/\$(RUN_ID)/condor_\$(ClusterId).\$(ProcId).$job.log
queue 1

EOF
    done
}

for job in "${SELECTED_JOBS[@]}"; do
    if ! task_file_for_job "$job" >/dev/null; then
        echo "ERROR: unknown job '$job'" >&2
        usage >&2
        exit 2
    fi
done

if [[ -z "${RUN_ID:-}" ]]; then
    next_run_index=1
    for parent in "$RESULTS_DIR" "$LOG_DIR"; do
        if [[ -d "$parent" ]]; then
            for path in "$parent"/run_[0-9][0-9][0-9]; do
                [[ -d "$path" ]] || continue
                name="$(basename "$path")"
                idx="${name#run_}"
                idx=$((10#$idx))
                if (( idx >= next_run_index )); then
                    next_run_index=$((idx + 1))
                fi
            done
        fi
    done

    RUN_ID="$(printf 'run_%03d' "$next_run_index")"
fi

if [[ ! "$RUN_ID" =~ ^run_[0-9][0-9][0-9]$ ]]; then
    echo "ERROR: RUN_ID must look like run_001; got '$RUN_ID'" >&2
    exit 2
fi

mkdir -p "$LOG_DIR/$RUN_ID"

echo "Submitting MCMC parameter scan jobs with RUN_ID=$RUN_ID"
if [[ ${#SELECTED_JOBS[@]} -gt 0 ]]; then
    echo "  jobs:    ${SELECTED_JOBS[*]}"
fi
echo "  logs:    $LOG_DIR/$RUN_ID"
echo "  results: $RESULTS_DIR/$RUN_ID"

SUBMIT_TO_USE="$SUBMIT_FILE"
if [[ ${#SELECTED_JOBS[@]} -gt 0 ]]; then
    write_selected_submit_file
    SUBMIT_TO_USE="$TEMP_SUBMIT_FILE"
fi

condor_submit \
    HIC_BAYES_ROOT="$HIC_BAYES_ROOT" \
    PYTHON="$PYTHON" \
    RUN_ID="$RUN_ID" \
    "$SUBMIT_TO_USE"
