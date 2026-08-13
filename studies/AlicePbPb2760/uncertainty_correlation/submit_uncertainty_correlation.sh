#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIC_BAYES_ROOT="${HIC_BAYES_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"

ENV_FILE="$HIC_BAYES_ROOT/config/env.sh"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: missing env file: $ENV_FILE" >&2
    exit 2
fi

source "$ENV_FILE"

if [[ -z "${HIC_BAYES_RESULTS:-}" ]]; then
    echo "ERROR: HIC_BAYES_RESULTS is not set. Define it in $ENV_FILE" >&2
    exit 2
fi

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$HIC_BAYES_ROOT/.venv/bin/python" ]]; then
        PYTHON="$HIC_BAYES_ROOT/.venv/bin/python"
    else
        PYTHON="$(command -v python3 || true)"
    fi
fi

if [[ -z "$PYTHON" ]]; then
    cat >&2 <<EOF
ERROR: Python interpreter not found.

Activate the hic-bayes virtual environment first:

    source .venv/bin/activate

or define PYTHON explicitly:

    export PYTHON=/path/to/python
EOF
    exit 2
fi

ANALYSIS_DIR="${SCRIPT_DIR#$HIC_BAYES_ROOT/}"
SYSTEM_NAME="$(basename "$(dirname "$ANALYSIS_DIR")")"
STUDY_NAME="$(basename "$ANALYSIS_DIR")"

RUN_ROOT="$HIC_BAYES_RESULTS/$SYSTEM_NAME/$STUDY_NAME"
RESULTS_DIR="$RUN_ROOT/chains"
LOG_DIR="$RUN_ROOT/logs"
PLOTS_DIR="$RUN_ROOT/plots"

SUBMIT_FILE="$SCRIPT_DIR/scripts/condor_uncertainty_correlation.sub"
MANIFEST_FILE="$SCRIPT_DIR/manifest.yaml"

SELECTED_JOBS=()
RUN_DESCRIPTION=""
TEMP_SUBMIT_FILE=""

cleanup() {
    if [[ -n "$TEMP_SUBMIT_FILE" && -f "$TEMP_SUBMIT_FILE" ]]; then
        rm -f "$TEMP_SUBMIT_FILE"
    fi
}
trap cleanup EXIT

usage() {
    cat <<EOF
Usage: $(basename "$0") --description "description" [--run-id run_001] [--only ro_2.0_Grad_1]

Options:
  --description TEXT  Description saved in manifest.yaml.
  --run-id ID         Submit into an explicit run directory instead of auto-numbering.
  --only JOB          Submit only one job. May be repeated.
  -h, --help          Show this help.

Environment:
  HIC_BAYES_RESULTS must be defined in config/env.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --description|-d)
            [[ $# -ge 2 ]] || { echo "ERROR: --description requires a value" >&2; exit 2; }
            RUN_DESCRIPTION="$2"
            shift 2
            ;;
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

if [[ -z "$RUN_DESCRIPTION" ]]; then
    echo "ERROR: --description is required" >&2
    usage >&2
    exit 2
fi

task_file_for_job() {
    case "$1" in
        baseline_Grad_1) echo "uncertainty_correlation_baseline_PbPb2760_Grad_1.yaml" ;;
        baseline_Grad_2) echo "uncertainty_correlation_baseline_PbPb2760_Grad_2.yaml" ;;
        baseline_zeroexp_Grad_1) echo "uncertainty_correlation_zeroexp_baseline_PbPb2760_Grad_1.yaml" ;;
        baseline_zeroexp_Grad_2) echo "uncertainty_correlation_zeroexp_baseline_PbPb2760_Grad_2.yaml" ;;
        ro_0.0_Grad_1) echo "uncertainty_correlation_rbf_correlation_ro=0.0_PbPb2760_Grad_1.yaml" ;;
        ro_0.0_Grad_2) echo "uncertainty_correlation_rbf_correlation_ro=0.0_PbPb2760_Grad_2.yaml" ;;
        ro_2.0_Grad_1) echo "uncertainty_correlation_rbf_correlation_ro=2.0_PbPb2760_Grad_1.yaml" ;;
        ro_2.0_Grad_2) echo "uncertainty_correlation_rbf_correlation_ro=2.0_PbPb2760_Grad_2.yaml" ;;
        ro_4.0_Grad_1) echo "uncertainty_correlation_rbf_correlation_ro=4.0_PbPb2760_Grad_1.yaml" ;;
        ro_4.0_Grad_2) echo "uncertainty_correlation_rbf_correlation_ro=4.0_PbPb2760_Grad_2.yaml" ;;
        ro_6.0_Grad_1) echo "uncertainty_correlation_rbf_correlation_ro=6.0_PbPb2760_Grad_1.yaml" ;;
        ro_6.0_Grad_2) echo "uncertainty_correlation_rbf_correlation_ro=6.0_PbPb2760_Grad_2.yaml" ;;
        ro_8.0_Grad_1) echo "uncertainty_correlation_rbf_correlation_ro=8.0_PbPb2760_Grad_1.yaml" ;;
        ro_8.0_Grad_2) echo "uncertainty_correlation_rbf_correlation_ro=8.0_PbPb2760_Grad_2.yaml" ;;
        ro_10.0_Grad_1) echo "uncertainty_correlation_rbf_correlation_ro=10.0_PbPb2760_Grad_1.yaml" ;;
        ro_10.0_Grad_2) echo "uncertainty_correlation_rbf_correlation_ro=10.0_PbPb2760_Grad_2.yaml" ;;
        rho_0.1_Grad_1) echo "uncertainty_correlation_equicorr_rho=0.1_PbPb2760_Grad_1.yaml" ;;
        rho_0.1_Grad_2) echo "uncertainty_correlation_equicorr_rho=0.1_PbPb2760_Grad_2.yaml" ;;
        rho_0.2_Grad_1) echo "uncertainty_correlation_equicorr_rho=0.2_PbPb2760_Grad_1.yaml" ;;
        rho_0.2_Grad_2) echo "uncertainty_correlation_equicorr_rho=0.2_PbPb2760_Grad_2.yaml" ;;
        rho_0.4_Grad_1) echo "uncertainty_correlation_equicorr_rho=0.4_PbPb2760_Grad_1.yaml" ;;
        rho_0.4_Grad_2) echo "uncertainty_correlation_equicorr_rho=0.4_PbPb2760_Grad_2.yaml" ;;
        rho_0.6_Grad_1) echo "uncertainty_correlation_equicorr_rho=0.6_PbPb2760_Grad_1.yaml" ;;
        rho_0.6_Grad_2) echo "uncertainty_correlation_equicorr_rho=0.6_PbPb2760_Grad_2.yaml" ;;
        rho_0.8_Grad_1) echo "uncertainty_correlation_equicorr_rho=0.8_PbPb2760_Grad_1.yaml" ;;
        rho_0.8_Grad_2) echo "uncertainty_correlation_equicorr_rho=0.8_PbPb2760_Grad_2.yaml" ;;
        rho_0.9_Grad_1) echo "uncertainty_correlation_equicorr_rho=0.9_PbPb2760_Grad_1.yaml" ;;
        rho_0.9_Grad_2) echo "uncertainty_correlation_equicorr_rho=0.9_PbPb2760_Grad_2.yaml" ;;
        *) return 1 ;;
    esac
}

for job in "${SELECTED_JOBS[@]}"; do
    if ! task_file_for_job "$job" >/dev/null; then
        echo "ERROR: unknown job '$job'" >&2
        exit 2
    fi
done

if [[ -z "${RUN_ID:-}" ]]; then
    next_run_index=1
    for parent in "$RESULTS_DIR" "$LOG_DIR" "$PLOTS_DIR"; do
        if [[ -d "$parent" ]]; then
            for path in "$parent"/run_[0-9][0-9][0-9]; do
                [[ -d "$path" ]] || continue
                idx="$(basename "$path")"
                idx="${idx#run_}"
                idx=$((10#$idx))
                (( idx >= next_run_index )) && next_run_index=$((idx + 1))
            done
        fi
    done
    RUN_ID="$(printf 'run_%03d' "$next_run_index")"
fi

if [[ ! "$RUN_ID" =~ ^run_[0-9][0-9][0-9]$ ]]; then
    echo "ERROR: RUN_ID must look like run_001; got '$RUN_ID'" >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR/$RUN_ID" "$LOG_DIR/$RUN_ID" "$PLOTS_DIR/$RUN_ID"

write_manifest_entry() {
    local git_commit git_branch created_at
    git_commit="$(git -C "$HIC_BAYES_ROOT" rev-parse HEAD)"
    git_branch="$(git -C "$HIC_BAYES_ROOT" branch --show-current || true)"
    [[ -z "$git_branch" ]] && git_branch="DETACHED_HEAD"
    created_at="$(date --iso-8601=seconds)"

    [[ -f "$MANIFEST_FILE" ]] || echo "runs:" > "$MANIFEST_FILE"

    cat >> "$MANIFEST_FILE" <<EOF
  - run_id: "$RUN_ID"
    created_at: "$created_at"
    system: "$SYSTEM_NAME"
    study: "$STUDY_NAME"
    description: "$RUN_DESCRIPTION"
    git_branch: "$git_branch"
    git_commit: "$git_commit"
    hic_bayes_root: "$HIC_BAYES_ROOT"
    output_root: "$RUN_ROOT"
    results_dir: "$RESULTS_DIR/$RUN_ID"
    logs_dir: "$LOG_DIR/$RUN_ID"
    plots_dir: "$PLOTS_DIR/$RUN_ID"
EOF
}

write_selected_submit_file() {
    TEMP_SUBMIT_FILE="$(mktemp "${TMPDIR:-/tmp}/hic_bayes_uncertainty_correlation.XXXXXX.sub")"

    cat > "$TEMP_SUBMIT_FILE" <<'EOF'
universe     = vanilla
getenv       = True
initialdir   = $(HIC_BAYES_ROOT)

ANALYSIS_DIR = $(ANALYSIS_DIR)
TASK_DIR     = $(ANALYSIS_DIR)/sims

executable   = $(PYTHON)
request_cpus   = 3
request_memory = 1500
request_disk   = 20000
requirements = !regexp("qgp2", TARGET.Machine) && !regexp("usphydro", TARGET.Machine)

environment  = "HIC_BAYES_NTHREADS=3 HIC_BAYES_RUN_ID=$(RUN_ID) HIC_BAYES_RESULTS_ROOT=$(HIC_BAYES_RESULTS) HIC_BAYES_RUN_ROOT=$(RUN_ROOT) HIC_BAYES_RESULTS_DIR=$(RESULTS_DIR)/$(RUN_ID) HIC_BAYES_LOG_DIR=$(LOG_DIR)/$(RUN_ID) HIC_BAYES_PLOTS_DIR=$(PLOTS_DIR)/$(RUN_ID)"

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

write_manifest_entry

echo "Submitting uncertainty correlation jobs"
echo "  RUN_ID:             $RUN_ID"
echo "  description:        $RUN_DESCRIPTION"
echo "  analysis dir:       $ANALYSIS_DIR"
echo "  HIC_BAYES_ROOT:     $HIC_BAYES_ROOT"
echo "  HIC_BAYES_RESULTS:  $HIC_BAYES_RESULTS"
echo "  logs:               $LOG_DIR/$RUN_ID"
echo "  results:            $RESULTS_DIR/$RUN_ID"
echo "  plots:              $PLOTS_DIR/$RUN_ID"
echo "  manifest:           $MANIFEST_FILE"

SUBMIT_TO_USE="$SUBMIT_FILE"
if [[ ${#SELECTED_JOBS[@]} -gt 0 ]]; then
    echo "  jobs:               ${SELECTED_JOBS[*]}"
    write_selected_submit_file
    SUBMIT_TO_USE="$TEMP_SUBMIT_FILE"
fi

condor_submit \
    HIC_BAYES_ROOT="$HIC_BAYES_ROOT" \
    HIC_BAYES_RESULTS="$HIC_BAYES_RESULTS" \
    ANALYSIS_DIR="$ANALYSIS_DIR" \
    RUN_ROOT="$RUN_ROOT" \
    RESULTS_DIR="$RESULTS_DIR" \
    LOG_DIR="$LOG_DIR" \
    PLOTS_DIR="$PLOTS_DIR" \
    PYTHON="$PYTHON" \
    RUN_ID="$RUN_ID" \
    "$SUBMIT_TO_USE"