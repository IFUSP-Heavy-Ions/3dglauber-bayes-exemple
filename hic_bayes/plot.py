"""Plot corner posteriors from a single config file.

This script can render individual plots, comparison plots, or both.
To change experiments, parameter groups, titles, colors, and output names,
edit the JSON config passed with --config.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


def _load_config(path: str) -> dict:
    config_path = Path(path)
    with config_path.open() as f:
        cfg = json.load(f)
    _hydrate_parameter_labels(cfg, config_path)
    _hydrate_parameter_groups(cfg)
    return cfg


def _task_config_candidates(config_path: Path) -> list[Path]:
    """Return YAML task configs that can define parameter labels for this plot config."""
    candidates: list[Path] = []
    plot_dir = config_path.parent
    study_dir = plot_dir.parent

    parameter_config = None
    try:
        with config_path.open() as f:
            parameter_config = json.load(f).get("global", {}).get("parameter_config")
    except (OSError, json.JSONDecodeError):
        parameter_config = None

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

    return candidates


def _hydrate_parameter_labels(cfg: dict, config_path: Path) -> None:
    """Use simulation YAMLs as the single source of truth for parameter labels."""
    for task_path in _task_config_candidates(config_path):
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
            cfg.setdefault("global", {})["parameter_config"] = str(
                task_path.relative_to(config_path.parent.parent)
                if task_path.is_relative_to(config_path.parent.parent)
                else task_path
            )
            return

    if cfg.get("parameter_labels"):
        return

    raise KeyError(
        "parameter_labels is not set and no simulation YAML with parameters was found. "
        "Set global.parameter_config or keep task YAMLs under the study sims/ directory."
    )


def _hydrate_parameter_groups(cfg: dict) -> None:
    if "parameter_groups" in cfg:
        return
    n_params = len(cfg.get("parameter_labels", []))
    split = n_params // 2
    cfg["parameter_groups"] = {
        "group1": list(range(0, split)),
        "group2": list(range(split, n_params)),
    }

def _output_formats(gcfg: dict) -> list[str]:
    configured = gcfg.get("formats", gcfg.get("format", "pdf"))
    if isinstance(configured, str):
        formats = [
            fmt.strip().lstrip(".").lower()
            for fmt in configured.split(",")
            if fmt.strip()
        ]
    else:
        formats = [
            str(fmt).strip().lstrip(".").lower()
            for fmt in configured
            if str(fmt).strip()
        ]

    if not formats:
        raise ValueError("No output formats configured. Use global.format or global.formats.")

    return list(dict.fromkeys(formats))

def _plot_backend(gcfg: dict) -> str:
    backend = str(gcfg.get("plot_backend", "seaborn")).strip().lower()
    if backend not in {"seaborn", "corner"}:
        raise ValueError("global.plot_backend must be 'seaborn' or 'corner'.")
    return backend

def _load_corner_module():
    try:
        import corner as corner_lib
    except ImportError as exc:
        raise ImportError(
            "global.plot_backend='corner' requires the 'corner' package. "
            "Install it in this environment or set global.plot_backend='seaborn'."
        ) from exc
    return corner_lib

def _latest_chain_csv(experiment_dir: Path) -> Path:
    direct_csv = experiment_dir / "chain.csv"
    if direct_csv.exists():
        return direct_csv

    run_dirs = sorted([p for p in experiment_dir.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No chain.csv or run_* folders found in: {experiment_dir}")

    for run_dir in reversed(run_dirs):
        csv_path = run_dir / "chain.csv"
        if csv_path.exists():
            return csv_path

    raise FileNotFoundError(f"No chain.csv found under run_* in: {experiment_dir}")


def _load_chain(
    path: Path,
    n_samples: int,
    prune: int,
    seed: int,
    n_params: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if n_params is not None:
        df = df.iloc[:, :n_params].copy()
    else:
        df = df.copy()
    if prune > 1:
        df = df.iloc[::prune].copy()

    if n_samples > 0 and len(df) > n_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(df), size=n_samples, replace=False))
        df = df.iloc[idx].copy()

    return df.reset_index(drop=True)


def _align_corner_axis_labels(g: sns.PairGrid) -> None:
    """Keep PairGrid edge labels centered across all corner subplots."""
    for ax in g.axes[-1, :]:
        if ax is not None:
            ax.xaxis.set_label_coords(0.5, -0.70)
            ax.xaxis.label.set_horizontalalignment("center")

    for ax in g.axes[:, 0]:
        if ax is not None:
            ax.yaxis.set_label_coords(-0.70, 0.5)
            ax.yaxis.label.set_verticalalignment("center")


def _add_corner_legend(
    fig: plt.Figure,
    labels: list[str],
    colors: list[str],
    title: str | None = None,
    fontsize: int = 9,
) -> Legend:
    handles = [
        Patch(facecolor=color, edgecolor=color, alpha=0.45, label=label)
        for label, color in zip(labels, colors)
    ]
    return fig.legend(
        handles=handles,
        title=title,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.92),
        frameon=True,
        fontsize=fontsize,
        title_fontsize=fontsize,
    )


def _style_corner_axis_text(
    g: sns.PairGrid,
    axis_label_fontsize: int | None,
    tick_label_fontsize: int | None,
) -> None:
    for row in g.axes:
        for ax in row:
            if ax is None:
                continue
            if axis_label_fontsize is not None:
                ax.xaxis.label.set_size(axis_label_fontsize)
                ax.yaxis.label.set_size(axis_label_fontsize)
            if tick_label_fontsize is not None:
                ax.tick_params(axis="both", labelsize=tick_label_fontsize)


def _style_matplotlib_corner_axes(
    fig: plt.Figure,
    tick_rotation: int,
    axis_label_fontsize: int | None,
    tick_label_fontsize: int | None,
) -> None:
    for ax in fig.axes:
        ax.tick_params(axis="x", labelrotation=tick_rotation, pad=2)
        ax.tick_params(axis="y", labelrotation=tick_rotation, pad=2)
        if axis_label_fontsize is not None:
            ax.xaxis.label.set_size(axis_label_fontsize)
            ax.yaxis.label.set_size(axis_label_fontsize)
        if tick_label_fontsize is not None:
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)


def _save_matplotlib_corner(
    fig: plt.Figure,
    out_path: Path,
    dpi: int,
    legend: Legend,
    title_artist=None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bbox_extra_artists = [legend]
    if title_artist is not None:
        bbox_extra_artists.append(title_artist)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        bbox_extra_artists=bbox_extra_artists,
    )
    plt.close(fig)


def _has_chain_data(path: Path) -> bool:
    return path.exists() and (
        (path / "chain.csv").exists()
        or any(
        child.is_dir() and child.name.startswith("run_") for child in path.iterdir()
        )
    )


def _resolve_under(root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _require_runtime_env(var_name: str) -> Path:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"{var_name} is not set. This plotting entrypoint requires the runtime "
            f"environment to define the mounted results/plots directories."
        )
    return Path(os.path.expanduser(os.path.expandvars(value)))


def _corner_plot_single(
    df: pd.DataFrame,
    param_labels: list[str],
    param_indices: list[int],
    title: str,
    out_path: Path,
    color: str,
    legend_label: str,
    legend_title: str,
    tick_rotation: int,
    dpi: int,
    font_scale: float,
    axis_label_fontsize: int | None,
    tick_label_fontsize: int | None,
    title_fontsize: int,
    legend_fontsize: int,
    plot_backend: str,
) -> None:
    if plot_backend == "corner":
        corner_lib = _load_corner_module()
        labels = [param_labels[i] for i in param_indices]
        data = df.iloc[:, param_indices].to_numpy()
        n_dim = len(labels)
        fig = plt.figure(figsize=(max(7.0, 1.55 * n_dim), max(7.0, 1.55 * n_dim)))
        fig = corner_lib.corner(
            data,
            labels=labels,
            fig=fig,
            color=color,
            fill_contours=True,
            plot_datapoints=False,
            plot_density=False,
            smooth=1.0,
            hist_kwargs={"density": True, "color": color, "linewidth": 1.6},
            contour_kwargs={"colors": color},
        )
        _style_matplotlib_corner_axes(
            fig,
            tick_rotation,
            axis_label_fontsize,
            tick_label_fontsize,
        )
        title_artist = fig.suptitle(title, y=0.995, fontsize=title_fontsize)
        fig.tight_layout(rect=[0.08, 0.08, 0.92, 0.94])
        legend = _add_corner_legend(
            fig,
            [legend_label],
            [color],
            title=legend_title,
            fontsize=legend_fontsize,
        )
        _save_matplotlib_corner(fig, out_path, dpi=dpi, legend=legend, title_artist=title_artist)
        return

    sns.set_context("notebook", font_scale=font_scale)
    sns.set_style("ticks")

    labels = [param_labels[i] for i in param_indices]
    plot_df = df.iloc[:, param_indices].copy()
    plot_df.columns = labels

    n_dim = len(labels)
    height = max(1.55, 14.0 / max(1, n_dim))

    g = sns.PairGrid(plot_df, vars=labels, corner=True, diag_sharey=False, height=height)
    g.map_lower(
        sns.kdeplot,
        fill=True,
        levels=5,
        thresh=0.06,
        alpha=0.35,
        color=color,
    )
    g.map_diag(
        sns.kdeplot,
        fill=True,
        alpha=0.45,
        linewidth=1.6,
        color=color,
    )

    # Match requested styling: rotate x/y tick labels by 45 degrees.
    for row in g.axes:
        for ax in row:
            if ax is None:
                continue
            ax.tick_params(axis="x", labelrotation=tick_rotation, pad=2)
            ax.tick_params(axis="y", labelrotation=tick_rotation, pad=2)

    _align_corner_axis_labels(g)
    _style_corner_axis_text(g, axis_label_fontsize, tick_label_fontsize)
    title_artist = g.figure.suptitle(title, y=0.995, fontsize=title_fontsize)
    g.figure.tight_layout(rect=[0.2, 0.2, 1.0, 0.94])
    legend = _add_corner_legend(
        g.figure,
        [legend_label],
        [color],
        title=legend_title,
        fontsize=legend_fontsize,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        bbox_extra_artists=[legend, title_artist],
    )
    plt.close(g.figure)


def _corner_plot_compare(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    param_labels: list[str],
    param_indices: list[int],
    title: str,
    out_path: Path,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
    legend_title: str,
    tick_rotation: int,
    dpi: int,
    font_scale: float,
    axis_label_fontsize: int | None,
    tick_label_fontsize: int | None,
    title_fontsize: int,
    legend_fontsize: int,
    plot_backend: str,
) -> None:
    _corner_plot_multi(
        datasets=[
            (left_df, left_label, left_color),
            (right_df, right_label, right_color),
        ],
        param_labels=param_labels,
        param_indices=param_indices,
        title=title,
        out_path=out_path,
        legend_title=legend_title,
        tick_rotation=tick_rotation,
        dpi=dpi,
        font_scale=font_scale,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        title_fontsize=title_fontsize,
        legend_fontsize=legend_fontsize,
        plot_backend=plot_backend,
    )


def _corner_plot_multi(
    datasets: list[tuple[pd.DataFrame, str, str]],
    param_labels: list[str],
    param_indices: list[int],
    title: str,
    out_path: Path,
    legend_title: str,
    tick_rotation: int,
    dpi: int,
    font_scale: float,
    axis_label_fontsize: int | None,
    tick_label_fontsize: int | None,
    title_fontsize: int,
    legend_fontsize: int,
    plot_backend: str,
) -> None:
    if plot_backend == "corner":
        corner_lib = _load_corner_module()
        labels = [param_labels[i] for i in param_indices]
        n_dim = len(labels)
        fig = plt.figure(figsize=(max(7.0, 1.55 * n_dim), max(7.0, 1.55 * n_dim)))
        for df, _label, color in datasets:
            fig = corner_lib.corner(
                df.iloc[:, param_indices].to_numpy(),
                labels=labels,
                fig=fig,
                color=color,
                fill_contours=True,
                plot_datapoints=False,
                plot_density=False,
                smooth=1.0,
                hist_kwargs={"density": True, "color": color, "linewidth": 1.4},
                contour_kwargs={"colors": color},
            )
        _style_matplotlib_corner_axes(
            fig,
            tick_rotation,
            axis_label_fontsize,
            tick_label_fontsize,
        )
        title_artist = fig.suptitle(title, y=0.995, fontsize=title_fontsize)
        fig.tight_layout(rect=[0.08, 0.08, 0.86, 0.94])
        legend = _add_corner_legend(
            fig,
            [label for _, label, _ in datasets],
            [color for _, _, color in datasets],
            title=legend_title,
            fontsize=legend_fontsize,
        )
        _save_matplotlib_corner(fig, out_path, dpi=dpi, legend=legend, title_artist=title_artist)
        return

    sns.set_context("notebook", font_scale=font_scale)
    sns.set_style("ticks")

    labels = [param_labels[i] for i in param_indices]
    frames = []
    palette = {}
    for df, label, color in datasets:
        data = df.iloc[:, param_indices].copy()
        data.columns = labels
        data[legend_title] = label
        frames.append(data)
        palette[label] = color

    combo = pd.concat(frames, ignore_index=True)

    n_dim = len(labels)
    height = max(1.55, 14.0 / max(1, n_dim))

    g = sns.PairGrid(
        combo,
        vars=labels,
        hue=legend_title,
        corner=True,
        diag_sharey=False,
        height=height,
        palette=palette,
    )
    g.map_lower(
        lambda x, y, color, **kwargs: sns.kdeplot(
            x=x,
            y=y,
            fill=True,
            levels=5,
            thresh=0.06,
            alpha=0.28,
            color=color,
            **kwargs,
        )
    )
    g.map_diag(
        sns.kdeplot,
        fill=True,
        alpha=0.40,
        linewidth=1.6,
    )
    for row in g.axes:
        for ax in row:
            if ax is None:
                continue
            ax.tick_params(axis="x", labelrotation=tick_rotation, pad=2)
            ax.tick_params(axis="y", labelrotation=tick_rotation, pad=2)

    _align_corner_axis_labels(g)
    _style_corner_axis_text(g, axis_label_fontsize, tick_label_fontsize)
    title_artist = g.figure.suptitle(title, y=0.995, fontsize=title_fontsize)
    g.figure.tight_layout(rect=[0.2, 0.2, 0.86, 0.94])
    legend = _add_corner_legend(
        g.figure,
        [label for _, label, _ in datasets],
        [color for _, _, color in datasets],
        title=legend_title,
        fontsize=legend_fontsize,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        bbox_extra_artists=[legend, title_artist],
    )
    plt.close(g.figure)


def _resolve_exp_dir(results_root: Path, rel_dir: str) -> Path:
    rel_path = Path(rel_dir)

    if rel_path.is_absolute() or _has_chain_data(rel_path):
        return rel_path

    rel_parts = rel_path.parts
    if "results" in rel_parts:
        # Legacy configs may store full paths like config/.../results/run_001/...
        # while results_root already points to .../results or .../results/run_001.
        idx = rel_parts.index("results")
        tail = Path(*rel_parts[idx + 1 :])
        if tail.parts and results_root.name.startswith("run_") and tail.parts[0] == results_root.name:
            tail = Path(*tail.parts[1:])
        normalized = results_root / tail
        if _has_chain_data(normalized):
            return normalized

    direct_candidates = [results_root / rel_path]
    if rel_path.parts and rel_path.parts[0] == results_root.name:
        direct_candidates.append(results_root.parent / rel_path)

    for direct in direct_candidates:
        if _has_chain_data(direct):
            return direct

    leaf = Path(rel_dir).name
    candidates = [p for p in results_root.rglob(leaf) if p.is_dir()]
    for cand in candidates:
        if _has_chain_data(cand):
            return cand
    return direct_candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate individual and/or comparison corner plots from config file."
    )
    parser.add_argument("--config", required=True, help="JSON config file path.")
    parser.add_argument(
        "--mode",
        choices=["individual", "comparison", "all"],
        default="comparison",
        help="Which plot jobs to run from config.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    gcfg = cfg["global"]
    param_labels = cfg["parameter_labels"]
    groups = cfg["parameter_groups"]
    experiments = cfg["experiments"]
    individual_jobs = cfg["individual_plots"].get("jobs", [])
    comparison_jobs = cfg.get("comparison_plots", {}).get("jobs", [])

    analysis_root = Path(gcfg.get("analysis_root", "."))
    results_root = _require_runtime_env("HIC_BAYES_RESULTS_DIR")
    out_root = _require_runtime_env("HIC_BAYES_PLOTS_DIR")
    out_individual = out_root / cfg["individual_plots"].get("output_subdir", "individual")
    out_comparison = out_root / cfg.get("comparison_plots", {}).get("output_subdir", "comparisons")
    n_samples = int(gcfg.get("n_samples", 1400))
    prune = int(gcfg.get("prune", 1))
    seed = int(gcfg.get("seed", 42))
    formats = _output_formats(gcfg)
    plot_backend = _plot_backend(gcfg)
    tick_rotation = int(gcfg.get("tick_rotation", 45))
    dpi = int(gcfg.get("dpi", 300))
    font_scale = float(gcfg.get("font_scale", 1.15))
    axis_label_fontsize = gcfg.get("axis_label_fontsize", 8)
    if axis_label_fontsize is not None:
        axis_label_fontsize = int(axis_label_fontsize)
    tick_label_fontsize = gcfg.get("tick_label_fontsize", 6)
    if tick_label_fontsize is not None:
        tick_label_fontsize = int(tick_label_fontsize)
    title_fontsize = int(gcfg.get("title_fontsize", 11))
    legend_fontsize = int(gcfg.get("legend_fontsize", 9))
    legend_title = str(gcfg.get("legend_title", "Experiment"))

    if args.mode in ("individual", "all"):
        out_individual.mkdir(parents=True, exist_ok=True)
    if args.mode in ("comparison", "all"):
        out_comparison.mkdir(parents=True, exist_ok=True)

    print("Loading chains for corner plots...")
    loaded: dict[str, pd.DataFrame] = {}
    for exp_name, exp_cfg in experiments.items():
        rel_dir = exp_cfg["path"]
        exp_dir = _resolve_exp_dir(results_root, rel_dir)
        print(f"  resolving {exp_name:<30} path={rel_dir} -> {exp_dir}")
        chain_csv = _latest_chain_csv(exp_dir)
        df = _load_chain(
            chain_csv,
            n_samples=n_samples,
            prune=prune,
            seed=seed,
            n_params=len(param_labels),
        )
        loaded[exp_name] = df
        print(f"  {exp_name:<30} -> {len(df):>5} samples from {chain_csv}")

    if args.mode in ("individual", "all"):
        print("\nGenerating individual corner plots from config jobs...")
        for job in individual_jobs:
            exp_name = job["experiment"]
            exp_label = experiments.get(exp_name, {}).get("label", exp_name)
            output_prefix = job.get("output_prefix", f"corner_{exp_name}")
            title_prefix = job.get("title_prefix", f"Uncertainty correlation: {exp_label}")
            legend_label = job.get("legend_label", exp_label)
            color = job.get("color", "#1f77b4")
            group_names = job.get("groups", ["group1", "group2"])

            if exp_name not in loaded:
                raise KeyError(f"Experiment {exp_name!r} is not defined in experiments.")

            for idx, group_name in enumerate(group_names, start=1):
                if group_name not in groups:
                    raise KeyError(f"Group {group_name!r} is not defined in parameter_groups.")
                param_idx = groups[group_name]
                for fmt in formats:
                    out_path = out_individual / f"{output_prefix}_{idx}.{fmt}"
                    _corner_plot_single(
                        loaded[exp_name],
                        param_labels,
                        param_idx,
                        title_prefix,
                        out_path,
                        color=color,
                        legend_label=legend_label,
                        legend_title=legend_title,
                        tick_rotation=tick_rotation,
                        dpi=dpi,
                        font_scale=font_scale,
                        axis_label_fontsize=axis_label_fontsize,
                        tick_label_fontsize=tick_label_fontsize,
                        title_fontsize=title_fontsize,
                        legend_fontsize=legend_fontsize,
                        plot_backend=plot_backend,
                    )
                    print(f"  saved: {out_path}")

    if args.mode in ("comparison", "all"):
        print("\nGenerating comparison corner plots from config jobs...")
        for job in comparison_jobs:
            if "experiments" in job:
                entries = []
                for exp in job["experiments"]:
                    if isinstance(exp, str):
                        exp_key = exp
                        exp_cfg = experiments.get(exp_key, {})
                        entries.append(
                            {
                                "experiment": exp_key,
                                "label": exp_cfg.get("label", exp_key),
                            }
                        )
                    else:
                        entries.append(exp)
            else:
                left_key = job["left_experiment"]
                right_key = job["right_experiment"]
                entries = [
                    {
                        "experiment": left_key,
                        "label": job.get(
                            "left_label",
                            experiments.get(left_key, {}).get("label", left_key),
                        ),
                        "color": job.get("left_color", "#1f77b4"),
                    },
                    {
                        "experiment": right_key,
                        "label": job.get(
                            "right_label",
                            experiments.get(right_key, {}).get("label", right_key),
                        ),
                        "color": job.get("right_color", "#d62728"),
                    },
                ]

            default_colors = [
                "#4c78a8",
                "#f58518",
                "#e45756",
                "#72b7b2",
                "#54a24b",
                "#b279a2",
                "#ff9da6",
                "#9d755d",
            ]
            datasets = []
            exp_keys = []
            labels_for_title = []
            for idx_entry, entry in enumerate(entries):
                exp_key = entry["experiment"]
                if exp_key not in loaded:
                    raise KeyError(f"Comparison references unknown experiment: {exp_key!r}.")
                exp_keys.append(exp_key)
                label = entry.get("label", experiments.get(exp_key, {}).get("label", exp_key))
                labels_for_title.append(label)
                color = entry.get("color", default_colors[idx_entry % len(default_colors)])
                datasets.append((loaded[exp_key], label, color))

            if len(datasets) < 2:
                raise ValueError("Comparison jobs require at least two experiments.")

            output_prefix = job.get("output_prefix", f"corner_{'_vs_'.join(exp_keys)}")
            title_prefix = job.get("title_prefix", " vs ".join(labels_for_title))
            group_names = job.get("groups", ["group1", "group2"])

            for idx, group_name in enumerate(group_names, start=1):
                if group_name not in groups:
                    raise KeyError(f"Group {group_name!r} is not defined in parameter_groups.")
                param_idx = groups[group_name]
                for fmt in formats:
                    out_path = out_comparison / f"{output_prefix}_{idx}.{fmt}"
                    _corner_plot_multi(
                        datasets,
                        param_labels,
                        param_idx,
                        title_prefix,
                        out_path,
                        legend_title=legend_title,
                        tick_rotation=tick_rotation,
                        dpi=dpi,
                        font_scale=font_scale,
                        axis_label_fontsize=axis_label_fontsize,
                        tick_label_fontsize=tick_label_fontsize,
                        title_fontsize=title_fontsize,
                        legend_fontsize=legend_fontsize,
                        plot_backend=plot_backend,
                    )
                    print(f"  saved: {out_path}")

    print("\nDone.")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
