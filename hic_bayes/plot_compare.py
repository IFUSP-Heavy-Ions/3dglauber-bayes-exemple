"""Small CLI wrapper for run-aware corner-plot comparisons.

Usage examples from the hic-bayes root:

    python3 hic_bayes/plot_compare.py AtlasPbPb2760/baseline run_002
    python3 hic_bayes/plot_compare.py AtlasPbPb2760/baseline run_002 run_003
    python3 hic_bayes/plot_compare.py AtlasPbPb2760/baseline run_002 --compare
    python3 hic_bayes/plot_compare.py AtlasPbPb2760/emulator_params run_001 --all

The first argument is the experiment/study directory.  The wrapper loads
``corner_plot_config.json`` from that directory, builds a temporary config for
the requested run mode, and delegates the actual plotting to ``hic_bayes/plot.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Any

import yaml


DEFAULT_COLORS = (
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _hydrate_parameter_labels(cfg: dict[str, Any], config_path: Path) -> None:
    """Use simulation YAMLs as the single source of truth for parameter labels."""
    study_dir = config_path.parent.parent
    candidates: list[Path] = []
    parameter_config = cfg.get("global", {}).get("parameter_config")
    if parameter_config:
        parameter_path = Path(str(parameter_config))
        if not parameter_path.is_absolute():
            parameter_path = study_dir / parameter_path
        candidates.append(parameter_path)

    sims_dir = study_dir / "sims"
    if sims_dir.exists():
        candidates.extend(
            path for path in sorted(sims_dir.glob("*.yaml"))
            if not path.name.startswith("._")
        )

    for task_path in candidates:
        if not task_path.exists():
            continue
        with task_path.open() as f:
            task_cfg = yaml.safe_load(f)
        params = task_cfg.get("parameters", []) if isinstance(task_cfg, dict) else []
        if params:
            cfg["parameter_labels"] = [
                str(param.get("label", param["name"])).replace("\\\\", "\\")
                for param in params
            ]
            cfg.setdefault("global", {})["parameter_config"] = str(task_path.resolve())
            return

    if cfg.get("parameter_labels"):
        return

    raise KeyError(
        "parameter_labels is not set and no simulation YAML with parameters was found. "
        "Set global.parameter_config or keep task YAMLs under the study sims/ directory."
    )


def _hydrate_parameter_groups(cfg: dict[str, Any]) -> None:
    if "parameter_groups" in cfg:
        return
    n_params = len(cfg.get("parameter_labels", []))
    split = n_params // 2
    cfg["parameter_groups"] = {
        "group1": list(range(0, split)),
        "group2": list(range(split, n_params)),
    }


def _resolve_config_path(study_dir: str | Path) -> Path:
    study_path = Path(study_dir)
    candidates = [
        study_path / "plot/corner_plot_config.json",
        Path("studies") / study_path / "plot/corner_plot_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find corner_plot_config.json for experiment/study directory. "
        f"Tried:\n  {tried}"
    )


def _normalize_run(run: str) -> str:
    run = str(run).strip()
    if run.startswith("run_"):
        return run
    if run.isdigit():
        return f"run_{int(run):03d}"
    return run


def _run_relative_path(path: str, run: str) -> str:
    rel = PurePosixPath(path)
    parts = rel.parts
    if parts and parts[0] == "results":
        parts = parts[1:]
    return str(PurePosixPath(run, *parts))


def _strip_results_prefix(path: str) -> str:
    """Return path relative to the nearest 'results' segment when present."""
    parts = list(PurePosixPath(str(path).strip()).parts)
    if not parts:
        return ""

    if parts[0] == ".":
        parts = parts[1:]

    if "results" in parts:
        idx = parts.index("results")
        parts = parts[idx + 1 :]

    return str(PurePosixPath(*parts)) if parts else ""


def _normalize_experiment_path_for_results_root(path: str, runs: list[str], results_root: Path) -> str:
    """Normalize legacy experiment paths so they resolve under HIC_BAYES_RESULTS_DIR."""
    normalized = _strip_results_prefix(path)
    if not normalized:
        return normalized

    parts = list(PurePosixPath(normalized).parts)
    if not parts:
        return normalized

    if parts[0].startswith("run_") and results_root.name.startswith("run_") and parts[0] == results_root.name:
        parts = parts[1:]

    return str(PurePosixPath(*parts)) if parts else ""


def _study_subdir(study_dir: str) -> Path:
    rel = PurePosixPath(str(study_dir).strip())
    parts = list(rel.parts)
    if parts and parts[0] == "studies":
        parts = parts[1:]
    return Path(*parts)


def _load_env_from_config(study_dir: str, global_cfg: dict[str, Any]) -> None:
    """Populate runtime paths from config/env.sh when they are not already set."""
    if os.environ.get("HIC_BAYES_RESULTS_DIR") and os.environ.get("HIC_BAYES_PLOTS_DIR"):
        return

    env_script = Path(__file__).resolve().parents[1] / "config" / "env.sh"
    if not env_script.exists():
        return

    if not Path("/storage").exists():
        raise RuntimeError(
            "/storage is not mounted. This HPC plotting workflow does not support "
            "Remote-HPC fallback paths."
        )

    results_base = f"/storage/{os.environ.get('USER', '')}/results/hic-bayes"

    if results_base:
        study_subdir = _study_subdir(study_dir)
        study_root = Path(results_base) / study_subdir
        results_subdir = str(global_cfg.get("results_root", "results")).strip() or "results"
        plots_subdir = str(global_cfg.get("output_root", "plots")).strip() or "plots"
        study_results_root = study_root / results_subdir
        study_plots_root = study_root / plots_subdir
        if not os.environ.get("HIC_BAYES_RESULTS_DIR"):
            os.environ["HIC_BAYES_RESULTS_DIR"] = str(study_results_root)
        if not os.environ.get("HIC_BAYES_PLOTS_DIR"):
            os.environ["HIC_BAYES_PLOTS_DIR"] = str(study_plots_root)


def _require_env_path(var_name: str) -> Path:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"{var_name} is not set. This plotting workflow requires the runtime "
            f"environment variables to point to the mounted results/plots trees."
        )
    return Path(os.path.expanduser(os.path.expandvars(value)))


def _has_chain_data(exp_dir: Path) -> bool:
    if not exp_dir.exists() or not exp_dir.is_dir():
        return False
    if (exp_dir / "chain.csv").exists():
        return True
    return any(child.is_dir() and child.name.startswith("run_") for child in exp_dir.iterdir())


def _filter_missing_runtime_comparison_jobs(cfg: dict[str, Any], results_root: Path) -> dict[str, Any]:
    """Drop pairwise comparison jobs whose chains are not available under results_root."""
    jobs = cfg.get("comparison_plots", {}).get("jobs", [])
    if not jobs:
        return cfg

    experiments = cfg.get("experiments", {})
    referenced: set[str] = set()
    kept_jobs = []
    skipped = 0
    missing_logs: list[str] = []
    for job in jobs:
        referenced_keys: set[str] = set()

        if "experiments" in job:
            missing = False
            for entry in job.get("experiments", []):
                exp_key = entry if isinstance(entry, str) else entry.get("experiment")
                if not exp_key:
                    continue
                exp_rel = experiments.get(exp_key, {}).get("path")
                if not exp_rel:
                    missing = True
                    break
                exp_path = results_root / Path(str(exp_rel))
                if not _has_chain_data(exp_path):
                    missing = True
                    missing_logs.append(
                        f"  missing chain for {exp_key}: rel={exp_rel} -> abs={exp_path}"
                    )
                else:
                    referenced_keys.add(exp_key)

            if missing:
                skipped += 1
                continue

            kept_jobs.append(job)
            referenced.update(referenced_keys)
            continue

        left_key = job.get("left_experiment")
        right_key = job.get("right_experiment")
        if not left_key or not right_key:
            kept_jobs.append(job)
            continue

        left_rel = experiments.get(left_key, {}).get("path")
        right_rel = experiments.get(right_key, {}).get("path")
        if not left_rel or not right_rel:
            skipped += 1
            continue

        left_path = results_root / Path(str(left_rel))
        right_path = results_root / Path(str(right_rel))
        left_ok = _has_chain_data(left_path)
        right_ok = _has_chain_data(right_path)
        if left_ok and right_ok:
            kept_jobs.append(job)
            referenced.update({left_key, right_key})
        else:
            skipped += 1
            if not left_ok:
                missing_logs.append(
                    f"  missing chain for {left_key}: rel={left_rel} -> abs={left_path}"
                )
            if not right_ok:
                missing_logs.append(
                    f"  missing chain for {right_key}: rel={right_rel} -> abs={right_path}"
                )

    cfg.setdefault("comparison_plots", {})["jobs"] = kept_jobs

    cfg["experiments"] = {
        key: value
        for key, value in cfg.get("experiments", {}).items()
        if key in referenced
    }

    if skipped:
        print(f"Skipping {skipped} comparison jobs with missing chains under {results_root}")
        for line in missing_logs:
            print(line)
    if not kept_jobs:
        print("No comparison jobs available for this run after filtering missing chains.")
    return cfg


def _apply_runtime_paths(cfg: dict[str, Any], runs: list[str], compare_mode: bool, study_dir: str) -> dict[str, Any]:
    """Adjust generated config to runtime output locations when env vars are set.

    For single-run workflows launched from submit scripts, ``HIC_BAYES_RESULTS_DIR``
    points directly to ``.../results/run_###`` and ``HIC_BAYES_PLOTS_DIR`` to
    ``.../plots/run_###``. In this case we strip the synthetic ``run_###/`` prefix
    added by run-aware config generation and write comparison figures to
    ``comparison`` under the run plot directory.
    """
    _load_env_from_config(study_dir, cfg.get("global", {}))
    results_root = _require_env_path("HIC_BAYES_RESULTS_DIR")
    plots_root = _require_env_path("HIC_BAYES_PLOTS_DIR")

    cfg.setdefault("global", {})
    cfg["global"]["analysis_root"] = "."
    cfg["global"]["results_root"] = str(results_root)

    for exp_cfg in cfg.get("experiments", {}).values():
        path = str(exp_cfg.get("path", ""))
        exp_cfg["path"] = _normalize_experiment_path_for_results_root(
            path,
            runs,
            results_root,
        )

    if compare_mode:
        cfg = _filter_missing_runtime_comparison_jobs(cfg, results_root)

    cfg["global"]["analysis_root"] = "."
    cfg["global"]["output_root"] = str(plots_root)
    if compare_mode and len(runs) == 1:
        cfg.setdefault("comparison_plots", {})["output_subdir"] = f"{runs[0]}/comparison"

    return cfg


def _base_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "global": copy.deepcopy(cfg["global"]),
        "parameter_groups": copy.deepcopy(cfg["parameter_groups"]),
        "experiments": {},
        "individual_plots": {
            "output_subdir": cfg.get("individual_plots", {}).get("output_subdir", "individual"),
            "jobs": [],
        },
        "comparison_plots": {
            "output_subdir": cfg.get("comparison_plots", {}).get("output_subdir", "comparisons"),
            "jobs": [],
        },
    }


def _experiments_for_run(cfg: dict[str, Any], run: str) -> dict[str, Any]:
    out = {}
    for name, exp in cfg.get("experiments", {}).items():
        entry = copy.deepcopy(exp)
        entry["path"] = _run_relative_path(entry["path"], run)
        out[name] = entry
    return out


def _run_experiment_key(run: str, experiment: str) -> str:
    return f"{run}__{experiment}"


def _individual_config(cfg: dict[str, Any], run: str) -> dict[str, Any]:
    out = _base_config(cfg)
    out["experiments"] = _experiments_for_run(cfg, run)
    out["individual_plots"]["output_subdir"] = (
        f"{run}/{cfg.get('individual_plots', {}).get('output_subdir', 'individual')}"
    )
    out["individual_plots"]["jobs"] = copy.deepcopy(
        cfg.get("individual_plots", {}).get("jobs", [])
    )
    return out


def _comparison_config(cfg: dict[str, Any], left_run: str, right_run: str) -> dict[str, Any]:
    out = _base_config(cfg)
    experiments = cfg.get("experiments", {})
    individual_jobs = cfg.get("individual_plots", {}).get("jobs", [])
    jobs_by_experiment = {
        job["experiment"]: job
        for job in individual_jobs
        if "experiment" in job
    }
    experiment_names = [
        job["experiment"]
        for job in individual_jobs
        if "experiment" in job
    ] or list(experiments)

    out["comparison_plots"]["output_subdir"] = (
        f"{left_run}_vs_{right_run}/"
        f"{cfg.get('comparison_plots', {}).get('output_subdir', 'comparisons')}"
    )

    for idx, exp_name in enumerate(experiment_names):
        if exp_name not in experiments:
            raise KeyError(f"Individual job references unknown experiment: {exp_name!r}.")

        left_key = _run_experiment_key(left_run, exp_name)
        right_key = _run_experiment_key(right_run, exp_name)
        left_entry = copy.deepcopy(experiments[exp_name])
        right_entry = copy.deepcopy(experiments[exp_name])
        left_entry["path"] = _run_relative_path(left_entry["path"], left_run)
        right_entry["path"] = _run_relative_path(right_entry["path"], right_run)
        out["experiments"][left_key] = left_entry
        out["experiments"][right_key] = right_entry

        source_job = jobs_by_experiment.get(exp_name, {})
        label = experiments[exp_name].get("label", exp_name)
        output_prefix = source_job.get("output_prefix", f"corner_{exp_name}")
        title_prefix = source_job.get("title_prefix", label)
        out["comparison_plots"]["jobs"].append(
            {
                "left_experiment": left_key,
                "right_experiment": right_key,
                "left_label": left_run,
                "right_label": right_run,
                "left_color": DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
                "right_color": "#d62728",
                "output_prefix": f"{output_prefix}_{left_run}_vs_{right_run}",
                "title_prefix": title_prefix,
                "groups": source_job.get("groups", ["group1", "group2"]),
            }
        )

    return out


def _configured_comparison_config(
    cfg: dict[str, Any],
    left_run: str,
    right_run: str | None = None,
) -> dict[str, Any]:
    out = _base_config(cfg)
    experiments = cfg.get("experiments", {})
    source_jobs = [
        job
        for job in cfg.get("comparison_plots", {}).get("jobs", [])
        if "left_experiment" in job and "right_experiment" in job
    ]
    if not source_jobs:
        raise ValueError("No pairwise comparison jobs found in comparison_plots.jobs.")

    same_run = right_run is None
    if same_run:
        right_run = left_run
        out["comparison_plots"]["output_subdir"] = (
            f"{left_run}/{cfg.get('comparison_plots', {}).get('output_subdir', 'comparisons')}"
        )
    else:
        out["comparison_plots"]["output_subdir"] = (
            f"{left_run}_vs_{right_run}/"
            f"{cfg.get('comparison_plots', {}).get('output_subdir', 'comparisons')}"
        )

    for job in source_jobs:
        left_exp = job["left_experiment"]
        right_exp = job["right_experiment"]
        if left_exp not in experiments:
            raise KeyError(f"Comparison references unknown left experiment: {left_exp!r}.")
        if right_exp not in experiments:
            raise KeyError(f"Comparison references unknown right experiment: {right_exp!r}.")

        left_key = _run_experiment_key(left_run, left_exp)
        right_key = _run_experiment_key(right_run, right_exp)
        if left_key not in out["experiments"]:
            left_entry = copy.deepcopy(experiments[left_exp])
            left_entry["path"] = _run_relative_path(left_entry["path"], left_run)
            out["experiments"][left_key] = left_entry
        if right_key not in out["experiments"]:
            right_entry = copy.deepcopy(experiments[right_exp])
            right_entry["path"] = _run_relative_path(right_entry["path"], right_run)
            out["experiments"][right_key] = right_entry

        out_job = copy.deepcopy(job)
        out_job["left_experiment"] = left_key
        out_job["right_experiment"] = right_key
        run_suffix = left_run if same_run else f"{left_run}_vs_{right_run}"
        out_job["output_prefix"] = (
            f"{job.get('output_prefix', f'corner_{left_exp}_vs_{right_exp}')}"
            f"_{run_suffix}"
        )
        out_job["title_prefix"] = job.get("title_prefix", f"{left_exp} vs {right_exp}")
        out["comparison_plots"]["jobs"].append(out_job)

    return out


def _all_experiments_config(cfg: dict[str, Any], run: str) -> dict[str, Any]:
    out = _base_config(cfg)
    out["comparison_plots"]["output_subdir"] = (
        f"{run}/{cfg.get('comparison_plots', {}).get('output_subdir', 'comparisons')}"
    )

    source_jobs = [
        job
        for job in cfg.get("comparison_plots", {}).get("jobs", [])
        if "experiments" in job
    ]
    if not source_jobs:
        source_jobs = [
            {
                "experiments": [
                    {
                        "experiment": exp_name,
                        "label": exp.get("label", exp_name),
                        "color": DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
                    }
                    for idx, (exp_name, exp) in enumerate(cfg.get("experiments", {}).items())
                ],
                "output_prefix": f"corner_all_experiments_{run}",
                "title_prefix": "All experiments",
                "groups": ["group1", "group2"],
            }
        ]

    for job in source_jobs:
        out_job = copy.deepcopy(job)
        out_job["output_prefix"] = f"{out_job.get('output_prefix', 'corner_all_experiments')}_{run}"
        out_job["title_prefix"] = out_job.get("title_prefix", "All experiments")
        out_job["experiments"] = []

        for entry_idx, entry in enumerate(job["experiments"]):
            if isinstance(entry, str):
                exp_name = entry
                exp_entry = cfg["experiments"][exp_name]
                label = exp_entry.get("label", exp_name)
                color = DEFAULT_COLORS[entry_idx % len(DEFAULT_COLORS)]
            else:
                exp_name = entry["experiment"]
                exp_entry = cfg["experiments"][exp_name]
                label = entry.get("label", exp_entry.get("label", exp_name))
                color = entry.get("color", DEFAULT_COLORS[entry_idx % len(DEFAULT_COLORS)])

            loaded_entry = copy.deepcopy(exp_entry)
            loaded_entry["path"] = _run_relative_path(loaded_entry["path"], run)
            out["experiments"][exp_name] = loaded_entry
            out_job["experiments"].append(
                {
                    "experiment": exp_name,
                    "label": label,
                    "color": color,
                }
            )

        out["comparison_plots"]["jobs"].append(out_job)

    return out


def _run_plot(tmp_config: Path, mode: str) -> None:
    plot_script = Path(__file__).with_name("plot.py")
    cmd = [sys.executable, str(plot_script), "--config", str(tmp_config), "--mode", mode]
    subprocess.run(cmd, check=True)


def _write_and_run(cfg: dict[str, Any], dry_run: bool, mode: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
        tmp_path = Path(f.name)

    print(f"Temporary plot config: {tmp_path}")
    if dry_run:
        print(json.dumps(cfg, indent=2))
        return

    _run_plot(tmp_path, mode=mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate run-aware corner plots from a study corner_plot_config.json."
    )
    parser.add_argument(
        "study_dir",
        help=(
            "Experiment/study directory containing corner_plot_config.json, "
            "e.g. AtlasPbPb2760/baseline or AtlasPbPb2760/emulator_params."
        ),
    )
    parser.add_argument("runs", nargs="+", help="One run for individual plots, or two runs for comparisons.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="With one run, put all configured experiments in the same comparison plot.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Use pairwise comparison_plots jobs instead of individual jobs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print generated config without plotting.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    runs = [_normalize_run(run) for run in args.runs]

    if len(runs) > 2:
        parser.error("pass one run, or two runs for run-to-run comparison")
    if args.all and len(runs) != 1:
        parser.error("--all expects exactly one run")
    if args.all and args.compare:
        parser.error("--all and --compare are mutually exclusive")

    config_path = _resolve_config_path(args.study_dir)
    cfg = _load_json(config_path)
    _hydrate_parameter_labels(cfg, config_path)
    _hydrate_parameter_groups(cfg)

    if args.all:
        out_cfg = _all_experiments_config(cfg, runs[0])
        mode = "comparison"
    elif args.compare:
        out_cfg = _configured_comparison_config(
            cfg,
            runs[0],
            runs[1] if len(runs) == 2 else None,
        )
        mode = "comparison"
    elif len(runs) == 1:
        out_cfg = _individual_config(cfg, runs[0])
        mode = "individual"
    else:
        out_cfg = _comparison_config(cfg, runs[0], runs[1])
        mode = "comparison"

    out_cfg = _apply_runtime_paths(out_cfg, runs, compare_mode=(mode == "comparison"), study_dir=args.study_dir)

    _write_and_run(out_cfg, args.dry_run, mode=mode)


if __name__ == "__main__":
    main()
