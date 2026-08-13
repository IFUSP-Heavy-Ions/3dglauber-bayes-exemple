#!/usr/bin/env python3
"""Create LHC uncertainty-correlation study configs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SYSTEMS = {
    "AliceXeXe5440": {
        "short": "XeXe5440",
        "title": "ALICE Xe+Xe 5.44 TeV",
        "data_dir": "data/AliceXeXe5440/lhc_predictions",
        "cache_file": "emulator-Xe-Xe-5440-idf-0-npc-10.pkl",
    },
    "AlicePbPb5020": {
        "short": "PbPb5020",
        "title": "ALICE Pb+Pb 5.02 TeV",
        "data_dir": "data/AlicePbPb5020/lhc_predictions",
        "cache_file": "emulator-Pb-Pb-5020-idf-0-npc-10.pkl",
    },
}

LABELS = {
    "ylossParam4At2": "$y_{loss}(2)$",
    "ylossParam4At4": "$y_{loss}(4)$",
    "ylossParam4At6": "$y_{loss}(6)$",
    "ylossParam4var": "$\\sigma_{y loss}$",
    "remnant_energy_loss_fraction": "$f_{remnant}$",
    "shadowing_factor": "$S$",
    "tau_form_mean": "$\\tau_{form}$",
    "BG": "$B_G$",
    "string_source_sigma_x": "$\\sigma_x$",
    "string_source_sigma_eta": "$\\sigma_\\eta$",
    "stringTransverseShiftFrac": "$f_{shift}$",
    "shear_viscosity_3_eta_over_s_T_kink_in_GeV": "$T_{\\eta,kink}$",
    "shear_viscosity_3_eta_over_s_low_T_slope_in_GeV": "$a_{\\eta,low}$",
    "shear_viscosity_3_eta_over_s_high_T_slope_in_GeV": "$a_{\\eta,high}$",
    "shear_viscosity_3_eta_over_s_at_kink": "$(\\eta/s)_{kink}$",
    "bulk_viscosity_3_zeta_over_s_max": "$(\\zeta/s)_{max}$",
    "bulk_viscosity_3_zeta_over_s_T_peak_in_GeV": "$T_{\\zeta,c}$",
    "bulk_viscosity_3_zeta_over_s_width_in_GeV": "$w_{\\zeta}$",
    "bulk_viscosity_3_zeta_over_s_lambda_asymm": "$\\lambda_{\\zeta}$",
    "eps_switch": "$\\epsilon_{sw}$",
}


def read_params(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "name": row["param"],
            "min": float(row["min"]),
            "max": float(row["max"]),
            "label": LABELS.get(row["param"], row["param"]),
        }
        for row in rows
    ]


def read_observable_columns(data_dir: Path) -> list[str]:
    meta = data_dir / "conversion_metadata.yaml"
    if not meta.exists():
        return []
    columns: list[str] = []
    in_columns = False
    for line in meta.read_text().splitlines():
        if line.strip() == "observable_columns:":
            in_columns = True
            continue
        if in_columns:
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith("- "):
                columns.append(stripped[2:].strip().strip("'\""))
            elif not line.startswith(" "):
                break
    return columns


def quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return quote(str(value))


def write_yaml(path: Path, cfg: dict) -> None:
    lines: list[str] = []
    lines.append(f"system: {scalar(cfg['system'])}")
    lines.append(f"description: {scalar(cfg['description'])}")
    lines.append("")
    lines.append("data:")
    for key, value in cfg["data"].items():
        lines.append(f"  {key}: {scalar(value)}")
    lines.append("")
    lines.append("model:")
    for key, value in cfg["model"].items():
        lines.append(f"  {key}: {scalar(value)}")
    lines.append("")
    lines.append("parameters:")
    for param in cfg["parameters"]:
        lines.append(
            "  - { "
            + f"name: {param['name']}, min: {param['min']}, max: {param['max']}, label: {scalar(param['label'])}"
            + " }"
        )
    lines.append("")
    if cfg["conversion"]["observable_columns"]:
        lines.append("conversion:")
        lines.append("  observable_columns:")
        lines.extend(f"    - {scalar(col)}" for col in cfg["conversion"]["observable_columns"])
        lines.append("")
    lines.append("pca:")
    lines.append(f"  n_components: {cfg['pca']['n_components']}")
    lines.append("")
    lines.append("emulator:")
    for key, value in cfg["emulator"].items():
        lines.append(f"  {key}: {scalar(value)}")
    lines.append("")
    lines.append("jetscape:")
    for key, value in cfg["jetscape"].items():
        lines.append(f"  {key}: {scalar(value)}")
    lines.append("")
    lines.append("experimental_covariance:")
    for key, value in cfg["experimental_covariance"].items():
        lines.append(f"  {key}: {scalar(value)}")
    lines.append("")
    lines.append("mcmc:")
    for key, value in cfg["mcmc"].items():
        lines.append(f"  {key}: {scalar(value)}")
    lines.append("")
    lines.append("closure_test:")
    lines.append(f"  enabled: {scalar(cfg['closure_test']['enabled'])}")
    path.write_text("\n".join(lines) + "\n")


def task_cfg(
    *,
    system_name: str,
    info: dict,
    params: list[dict],
    observable_columns: list[str],
    grad_idx: int,
    mode: str,
    value: float | None,
    n_components: int,
    n_steps: int,
    burnin: int,
    n_walkers: int,
    n_temps: int,
    n_threads: int,
) -> tuple[str, str, dict]:
    short = info["short"]
    grad = f"Grad_{grad_idx}"
    if mode == "baseline":
        key = f"baseline_{grad}"
        filename = f"uncertainty_correlation_baseline_{short}_{grad}.yaml"
        exp_cov = {"mode": "diag", "corr_rho": 0.0}
        desc = "diagonal experimental covariance"
    elif mode == "zeroexp":
        key = f"baseline_zeroexp_{grad}"
        filename = f"uncertainty_correlation_zeroexp_baseline_{short}_{grad}.yaml"
        exp_cov = {"mode": "none", "corr_rho": 0.0}
        desc = "no experimental covariance"
    elif mode == "equicorr":
        key = f"rho_{value:.1f}_{grad}"
        filename = f"uncertainty_correlation_equicorr_rho={value:.1f}_{short}_{grad}.yaml"
        exp_cov = {"mode": "equicorr", "corr_rho": float(value)}
        desc = f"equicorrelated experimental covariance with rho {value:.1f}"
    elif mode == "rbf":
        key = f"ro_{value:.1f}_{grad}"
        filename = f"uncertainty_correlation_rbf_correlation_ro={value:.1f}_{short}_{grad}.yaml"
        exp_cov = {"mode": "exp", "corr_length": float(value), "corr_rho": 0.0}
        desc = f"exponential experimental covariance with correlation length {value:.1f}"
    else:
        raise ValueError(mode)

    cfg = {
        "system": f"{system_name}_uncertainty_correlation_{grad}_{mode}"
        if value is None
        else f"{system_name}_uncertainty_correlation_{grad}_{mode}_{value:.1f}",
        "description": f"{info['title']} {grad}; {desc}",
        "data": {
            "base_dir": info["data_dir"],
            "experiment": "experiment.csv",
            "exp_error": "exp_error.csv",
        },
        "model": {
            "base_dir": "model/RHICAuAu200/Au-Au-200",
            "design": "design_points.csv",
            "design_ranges": "design_ranges.csv",
            "simulation": "simulation.csv",
            "validation_design": "validation_design_points.csv",
            "validation_simulation": "validation_simulation.csv",
        },
        "parameters": params,
        "conversion": {"observable_columns": observable_columns},
        "pca": {"n_components": n_components},
        "emulator": {
            "train_in_hic_bayes": True,
            "n_restarts": 4,
            "n_jobs": 1,
            "cache_dir": f"${{HIC_BAYES_RESULTS}}/{system_name}/uncertainty_correlation/emulador",
            "cache_file": info["cache_file"],
            "allow_version_mismatch": True,
        },
        "jetscape": {
            "transform_multiplicities": True,
            "monotonic_first_three": True,
        },
        "experimental_covariance": exp_cov,
        "mcmc": {
            "sampler": "emcee",
            "output_root": "${HIC_BAYES_RESULTS}/results",
            "run_tag": key,
            "shared_run_dir": True,
            "n_walkers": n_walkers,
            "n_steps": n_steps,
            "n_temps": n_temps,
            "n_threads": n_threads,
            "burnin": burnin,
            "thin": 10,
            "seed": 41 + grad_idx,
        },
        "closure_test": {"enabled": False},
    }
    return key, filename, cfg


def write_submit_script(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
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
SUBMIT_FILE="$SCRIPT_DIR/scripts/condor_uncertainty_correlation.sub"

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

if [[ ${#SELECTED_JOBS[@]} -gt 0 ]]; then
    tmp="$(mktemp)"
    awk 'BEGIN{copy=1} /^queue JOB,TASK from /{print; print "("; copy=0; next} copy{print}' "$SUBMIT_FILE" > "$tmp"
    for job in "${SELECTED_JOBS[@]}"; do
        awk -v j="$job" '$1 == j {print}' "$SUBMIT_FILE" >> "$tmp"
    done
    echo ")" >> "$tmp"
    SUBMIT_FILE="$tmp"
fi

condor_submit \
  -append "HIC_BAYES_ROOT=$HIC_BAYES_ROOT" \
  -append "PYTHON=$PYTHON" \
  -append "ANALYSIS_DIR=$ANALYSIS_DIR" \
  -append "HIC_BAYES_RESULTS=$HIC_BAYES_RESULTS" \
  -append "RUN_ROOT=$RUN_ROOT" \
  -append "RESULTS_DIR=$RESULTS_DIR" \
  -append "LOG_DIR=$LOG_DIR" \
  -append "PLOTS_DIR=$PLOTS_DIR" \
  -append "RUN_ID=$RUN_ID" \
  "$SUBMIT_FILE"
"""
    )
    path.chmod(0o755)


def write_condor(path: Path, jobs: list[tuple[str, str]]) -> None:
    lines = [
        "universe     = vanilla",
        "getenv       = True",
        "initialdir   = $(HIC_BAYES_ROOT)",
        "TASK_DIR     = $(ANALYSIS_DIR)/sims",
        "executable   = $(PYTHON)",
        "request_cpus   = 3",
        "request_memory = 1500",
        "request_disk   = 20000",
        'environment  = "HIC_BAYES_NTHREADS=3 HIC_BAYES_RUN_ID=$(RUN_ID) HIC_BAYES_RESULTS_ROOT=$(HIC_BAYES_RESULTS) HIC_BAYES_RUN_ROOT=$(RUN_ROOT) HIC_BAYES_RESULTS_DIR=$(RESULTS_DIR)/$(RUN_ID) HIC_BAYES_LOG_DIR=$(LOG_DIR)/$(RUN_ID) HIC_BAYES_PLOTS_DIR=$(PLOTS_DIR)/$(RUN_ID)"',
        "arguments    = -m pipelines.run_jetscape_lhc_analysis $(TASK_DIR)/$(TASK)",
        "output       = $(LOG_DIR)/$(RUN_ID)/condor_$(ClusterId).$(ProcId).$(JOB).out",
        "error        = $(LOG_DIR)/$(RUN_ID)/condor_$(ClusterId).$(ProcId).$(JOB).err",
        "log          = $(LOG_DIR)/$(RUN_ID)/condor_$(ClusterId).$(ProcId).$(JOB).log",
        "queue JOB,TASK from (",
    ]
    lines.extend(f"{job} {task}" for job, task in jobs)
    lines.append(")")
    path.write_text("\n".join(lines) + "\n")


def write_plot_config(path: Path, params: list[dict], jobs: list[tuple[str, str]]) -> None:
    config = {
        "global": {
            "analysis_root": ".",
            "results_root": "results",
            "output_root": "plots",
        },
        "experiments": {
            job: {"path": f"results/{job}", "label": job}
            for job, _ in jobs
        },
        "individual_plots": {
            "output_subdir": "individual",
            "jobs": [
                {
                    "experiment": job,
                    "output_prefix": f"corner_{job}",
                    "title": job,
                }
                for job, _ in jobs
            ],
        },
        "comparison_plots": {"output_subdir": "comparisons", "jobs": []},
    }
    path.write_text(json.dumps(config, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", choices=sorted(SYSTEMS))
    parser.add_argument("--design-ranges", default="model/RHICAuAu200/Au-Au-200/design_ranges.csv")
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=175)
    parser.add_argument("--burnin", type=int, default=100)
    parser.add_argument("--n-walkers", type=int, default=200)
    parser.add_argument("--n-temps", type=int, default=20)
    parser.add_argument("--n-threads", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = args.system or sorted(SYSTEMS)
    params = read_params(Path(args.design_ranges))
    equicorr = [round(0.1 * idx, 1) for idx in range(1, 10)]
    rbf = [0.0, 0.1, 0.5, 0.9, 1.3, 1.7, 2.0, 4.0, 6.0, 8.0, 10.0]

    for system_name in systems:
        info = SYSTEMS[system_name]
        study = Path("studies") / system_name / "uncertainty_correlation"
        sims = study / "sims"
        scripts = study / "scripts"
        plot = study / "plot"
        sims.mkdir(parents=True, exist_ok=True)
        scripts.mkdir(parents=True, exist_ok=True)
        plot.mkdir(parents=True, exist_ok=True)

        observable_columns = read_observable_columns(Path(info["data_dir"]))
        jobs: list[tuple[str, str]] = []
        for grad_idx in (1, 2):
            specs = [("baseline", None), ("zeroexp", None)]
            specs.extend(("equicorr", rho) for rho in equicorr)
            specs.extend(("rbf", ro) for ro in rbf)
            for mode, value in specs:
                job, filename, cfg = task_cfg(
                    system_name=system_name,
                    info=info,
                    params=params,
                    observable_columns=observable_columns,
                    grad_idx=grad_idx,
                    mode=mode,
                    value=value,
                    n_components=args.n_components,
                    n_steps=args.n_steps,
                    burnin=args.burnin,
                    n_walkers=args.n_walkers,
                    n_temps=args.n_temps,
                    n_threads=args.n_threads,
                )
                write_yaml(sims / filename, cfg)
                jobs.append((job, filename))

        write_submit_script(study / "submit_uncertainty_correlation.sh")
        write_condor(scripts / "condor_uncertainty_correlation.sub", jobs)
        write_plot_config(plot / "corner_plot_config.json", params, jobs)
        print(f"{system_name}: wrote {len(jobs)} tasks under {study}")


if __name__ == "__main__":
    main()
