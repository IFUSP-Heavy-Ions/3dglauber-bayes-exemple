#!/usr/bin/env bash
set -euo pipefail

PARENT_CLUSTER="${1:?missing parent cluster id}"
EXPECTED_JOBS="${2:?missing expected job count}"

: "${HIC_BAYES_ROOT:?missing HIC_BAYES_ROOT}"
: "${PYTHON:?missing PYTHON}"
: "${ANALYSIS_DIR:?missing ANALYSIS_DIR}"
: "${HIC_BAYES_LOG_DIR:?missing HIC_BAYES_LOG_DIR}"
: "${HIC_BAYES_RESULTS_DIR:?missing HIC_BAYES_RESULTS_DIR}"
: "${HIC_BAYES_PLOTS_DIR:?missing HIC_BAYES_PLOTS_DIR}"
: "${HIC_BAYES_RUN_ID:?missing HIC_BAYES_RUN_ID}"

cd "$HIC_BAYES_ROOT"

echo "Monitoring parent Condor cluster: $PARENT_CLUSTER"
echo "Expected parent jobs: $EXPECTED_JOBS"
echo "Run: $HIC_BAYES_RUN_ID"
echo "Logs: $HIC_BAYES_LOG_DIR"
echo "Results: $HIC_BAYES_RESULTS_DIR"
echo "Plots: $HIC_BAYES_PLOTS_DIR"

if [[ "$PARENT_CLUSTER" != "plot-only" ]]; then
    condor_q_args=()
    if [[ -n "${PARENT_SCHEDD:-}" ]]; then
        condor_q_args+=( -name "$PARENT_SCHEDD" )
        echo "Parent schedd: $PARENT_SCHEDD"
    fi

    while true; do
        if ! queue_output="$(condor_q "${condor_q_args[@]}" "$PARENT_CLUSTER" -af ClusterId)"; then
            echo "ERROR: condor_q failed while checking parent cluster $PARENT_CLUSTER" >&2
            exit 1
        fi

        [[ -n "$queue_output" ]] || break
        still_running="$(wc -l <<< "$queue_output" | tr -d ' ')"
        echo "$(date -Is) parent cluster still in queue: $still_running jobs"
        sleep "${PLOT_MONITOR_INTERVAL:-300}"
    done

    sleep 10

    shopt -s nullglob
    logs=( "$HIC_BAYES_LOG_DIR"/condor_"$PARENT_CLUSTER".*.log )
    shopt -u nullglob

    if [[ "${#logs[@]}" -ne "$EXPECTED_JOBS" ]]; then
        echo "ERROR: expected $EXPECTED_JOBS parent logs, found ${#logs[@]}" >&2
        printf 'Found logs:\n' >&2
        printf '  %s\n' "${logs[@]}" >&2
        exit 1
    fi

    failed=0
    for log_file in "${logs[@]}"; do
        if ! grep -q "Normal termination (return value 0)" "$log_file"; then
            echo "ERROR: parent job did not finish cleanly: $log_file" >&2
            failed=1
        fi
    done

    if [[ "$failed" -ne 0 ]]; then
        echo "ERROR: at least one parent job failed; plot step will not run." >&2
        exit 1
    fi

    echo "All parent jobs completed successfully."
else
    echo "Plot-only mode: skipping parent cluster checks."
fi

echo "Generating configured plots..."
PLOT_CONFIG="$HIC_BAYES_LOG_DIR/corner_plot_config.runtime.json"
"$PYTHON" - "$HIC_BAYES_ROOT/$ANALYSIS_DIR/plot/corner_plot_config.json" "$PLOT_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])

with source.open() as f:
    cfg = json.load(f)

sims_dir = source.parent.parent / "sims"
task_paths = sorted(
    path for path in sims_dir.glob("*.yaml")
    if not path.name.startswith("._")
)
if task_paths:
    cfg.setdefault("global", {})["parameter_config"] = str(task_paths[0].resolve())

target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

individual_jobs = len(cfg.get("individual_plots", {}).get("jobs", []))
comparison_jobs = len(cfg.get("comparison_plots", {}).get("jobs", []))
print(
    "Runtime plot config prepared: "
    f"{individual_jobs} individual jobs, {comparison_jobs} comparison jobs"
)
PY

echo "Running comparison plot jobs declared in corner_plot_config.json..."
"$PYTHON" hic_bayes/plot.py --config "$PLOT_CONFIG" --mode comparison
echo "Plot step completed."
