"""Checkpoint diagnostics for Au+Au 200 emulator equivalence.

This script compares the legacy JETSCAPE emulator and the hic-bayes-trained
JETSCAPE-compatible emulator without running MCMC.  It writes evidence files
for the major checkpoints that can move a posterior:

    config/input tables -> preprocessing/covariance -> PCA -> GP emulator
    -> pointwise log likelihood

Example
-------
Run from the hic-bayes repository root:

    HIC_BAYES_RESULTS=/path/to/results/hic-bayes \
    python studies/AllDetectorsAuAu200/emulator/scripts/diagnose_auau200_emulator_equivalence.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger("diagnose_auau200")


DEFAULT_HIC_CONFIG = Path("studies/AllDetectorsAuAu200/emulator/sims/emulator_compare_AuAu200_hicbayes.yaml")
DEFAULT_JET_CONFIG = Path("studies/AllDetectorsAuAu200/emulator/sims/emulator_compare_AuAu200_jetscape.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def _analysis_dir(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "sims" else config_path.parent


def _expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(path))))


def _resolve_path(base: Path, path: str | Path) -> Path:
    expanded = _expand(path)
    unresolved = re.search(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*", str(expanded))
    if unresolved:
        raise ValueError(f"Unresolved environment variable in path: {path!r}")
    return expanded if expanded.is_absolute() else base / expanded


def _ensure_results_env(repo: Path) -> None:
    if os.environ.get("HIC_BAYES_RESULTS"):
        return
    for parent in [repo, *repo.parents]:
        candidate = parent / "results/hic-bayes"
        if candidate.exists():
            os.environ["HIC_BAYES_RESULTS"] = str(candidate)
            LOGGER.info("Inferred HIC_BAYES_RESULTS=%s", candidate)
            return


def _table_path(cfg: dict[str, Any], key: str, *, kind: str, config_path: Path) -> Path:
    sections = [cfg.get("data", {})] if kind == "data" else [cfg.get("model", {}), cfg.get("data", {})]
    for section in sections:
        if key not in section:
            continue
        value = _expand(section[key])
        if value.is_absolute():
            return value
        base_dir = _expand(section.get("base_dir", "."))
        base = base_dir if base_dir.is_absolute() else Path.cwd() / base_dir
        return base / value
    raise KeyError(f"Could not resolve {key!r} in {kind} sections.")


def _read_table(cfg: dict[str, Any], key: str, *, kind: str, config_path: Path) -> pd.DataFrame:
    return pd.read_csv(_table_path(cfg, key, kind=kind, config_path=config_path))


def _resolve_emulator_path(cfg: dict[str, Any], config_path: Path) -> Path:
    em_cfg = cfg.get("emulator", {})
    base = _analysis_dir(config_path)
    if "input_pkl" in em_cfg:
        return _resolve_path(base, em_cfg["input_pkl"])
    if "input_dill" in em_cfg:
        return _resolve_path(base, em_cfg["input_dill"])
    cache_dir = em_cfg.get("cache_dir", "emulator")
    cache_file = em_cfg.get("cache_file")
    if not cache_file:
        raise ValueError("emulator.cache_file or emulator.input_pkl/input_dill is required.")
    return _resolve_path(base, cache_dir) / cache_file


def _load_emulator(cfg: dict[str, Any], config_path: Path):
    source_root = cfg.get("jetscape", {}).get("source_root")
    if source_root:
        source_path = Path(os.path.expanduser(os.path.expandvars(str(source_root))))
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        source_path = source_path.resolve()
        if source_path.exists():
            sys.path.insert(0, str(source_path))
            LOGGER.info("Added JETSCAPE source root for legacy dill: %s", source_path)
        else:
            LOGGER.warning("Configured JETSCAPE source root does not exist: %s", source_path)

    path = _resolve_emulator_path(cfg, config_path)
    LOGGER.info("Loading emulator: %s", path)
    with path.open("rb") as handle:
        if path.suffix == ".dill":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emulator = dill.load(handle)
            for warning in caught:
                LOGGER.warning("Unpickle warning for %s: %s: %s", path, warning.category.__name__, warning.message)
        else:
            emulator = pickle.load(handle)
    LOGGER.info("Loaded emulator type: %s.%s", type(emulator).__module__, type(emulator).__name__)
    return emulator, path


def _columns(cfg: dict[str, Any], table: pd.DataFrame) -> list[str]:
    return list(cfg.get("conversion", {}).get("observable_columns") or table.columns)


def _apply_removed_design_points(cfg: dict[str, Any], design: pd.DataFrame, sim: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    removed = [int(x) for x in cfg.get("conversion", {}).get("design_remove_idx", [])]
    if not removed:
        return design.copy(), sim.copy()
    keep = np.ones(len(design), dtype=bool)
    keep[removed] = False
    return design.loc[keep].reset_index(drop=True), sim.loc[keep].reset_index(drop=True)


def _group_columns(columns: list[str]) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for col in columns:
        key = col.rsplit("__bin", 1)[0] if "__bin" in col else col
        groups.setdefault(key, []).append(col)
    return list(groups.items())


def _is_multiplicity(obs: str) -> bool:
    return any(token in obs for token in ("dN", "DN", "dET", "DET"))


def _compute_experiment_cov(
    obs1: str,
    obs2: str,
    dy1: np.ndarray,
    dy2: np.ndarray,
    mode: str,
    corr_rho: float,
    corr_length: float | None,
    transform_multiplicities: bool,
) -> np.ndarray:
    mode = mode.strip().lower()
    if mode in {"none", "zero", "nocov", "noexp", "absent"}:
        return np.zeros((dy1.size, dy2.size))
    if mode == "diag":
        if obs1 != obs2:
            return np.zeros((dy1.size, dy2.size))
        if transform_multiplicities and _is_multiplicity(obs1):
            return np.diag(np.log(dy1**2 + 1.0))
        return np.diag(dy1**2)
    if mode in {"equicorr", "constant_corr", "corr"}:
        if obs1 != obs2:
            return np.outer(dy1, dy2) * corr_rho
        corr = np.full((dy1.size, dy2.size), corr_rho)
        np.fill_diagonal(corr, 1.0)
        return np.outer(dy1, dy2) * corr
    if mode in {"exp", "exponential", "rbf", "rbf_correlation"}:
        if obs1 != obs2:
            return np.zeros((dy1.size, dy2.size))
        if corr_length is None:
            raise ValueError("experimental_covariance.corr_length is required for exp/rbf mode.")
        if corr_length == 0:
            if transform_multiplicities and _is_multiplicity(obs1):
                return np.diag(np.log(dy1**2 + 1.0))
            return np.diag(dy1**2)
        x1 = np.linspace(0.0, 1.0, dy1.size)
        x2 = np.linspace(0.0, 1.0, dy2.size)
        corr = np.exp(-np.subtract.outer(x1, x2) ** 2 / float(corr_length) ** 2)
        return np.outer(dy1, dy2) * corr
    raise ValueError(f"Unknown experimental covariance mode: {mode!r}")


@dataclass
class PreparedData:
    groups: list[tuple[str, list[str]]]
    y_exp: np.ndarray
    exp_cov: np.ndarray


def _prepare_data(cfg: dict[str, Any], config_path: Path) -> PreparedData:
    exp_df = _read_table(cfg, "experiment", kind="data", config_path=config_path)
    err_df = _read_table(cfg, "exp_error", kind="data", config_path=config_path)
    columns = _columns(cfg, exp_df)
    exp_df = exp_df.loc[:, columns]
    err_df = err_df.loc[:, columns]

    cov_cfg = cfg.get("experimental_covariance", {})
    mode = str(cov_cfg.get("mode", "diag"))
    corr_rho = float(cov_cfg.get("corr_rho", 0.0))
    corr_length = cov_cfg.get("corr_length")
    corr_length = None if corr_length is None else float(corr_length)
    transform_multiplicities = bool(cfg.get("jetscape", {}).get("transform_multiplicities", True))

    groups = _group_columns(columns)
    y_blocks: list[np.ndarray] = []
    dy_blocks: list[np.ndarray] = []
    for obs, cols in groups:
        y = exp_df.loc[0, cols].to_numpy(dtype=float)
        err = err_df.loc[0, cols].to_numpy(dtype=float)
        if transform_multiplicities and _is_multiplicity(obs):
            y_safe = np.maximum(y, 1e-300)
            y_blocks.append(np.log(y_safe))
            dy_blocks.append(err / y_safe)
        else:
            y_blocks.append(y)
            dy_blocks.append(err)

    y_exp = np.concatenate(y_blocks)
    exp_cov = np.empty((y_exp.size, y_exp.size))
    row = 0
    for i, (obs1, cols1) in enumerate(groups):
        col = 0
        for j, (obs2, cols2) in enumerate(groups):
            block = _compute_experiment_cov(
                obs1,
                obs2,
                dy_blocks[i],
                dy_blocks[j],
                mode,
                corr_rho,
                corr_length,
                transform_multiplicities,
            )
            exp_cov[row : row + len(cols1), col : col + len(cols2)] = block
            col += len(cols2)
        row += len(cols1)
    return PreparedData(groups=groups, y_exp=y_exp, exp_cov=exp_cov)


def _pca_attr(emulator: Any, attr: str):
    pca_state = getattr(emulator, "pca_state", None)
    if attr == "scaler_mean":
        return pca_state.scaler.mean_ if pca_state is not None else emulator.scaler.mean_
    if attr == "scaler_scale":
        return pca_state.scaler.scale_ if pca_state is not None else emulator.scaler.scale_
    if attr == "pca_components":
        return pca_state.pca.components_ if pca_state is not None else emulator.pca.components_
    if attr == "pca_explained_variance":
        return pca_state.pca.explained_variance_ if pca_state is not None else emulator.pca.explained_variance_
    if attr == "pca_explained_variance_ratio":
        return pca_state.pca.explained_variance_ratio_ if pca_state is not None else emulator.pca.explained_variance_ratio_
    if attr == "transform_matrix":
        return pca_state.transform_matrix if pca_state is not None else emulator._trans_matrix
    if attr == "cov_trunc":
        return pca_state.cov_trunc if pca_state is not None else emulator._cov_trunc
    raise KeyError(attr)


def _observable_names(emulator: Any) -> list[str]:
    return list(getattr(emulator, "observables"))


def _flatten_prediction(emulator: Any, groups: list[tuple[str, list[str]]], theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_mean, pred_cov = emulator.predict(theta.reshape(1, -1), return_cov=True)
    mean_blocks: list[np.ndarray] = []
    n_obs = sum(len(cols) for _, cols in groups)
    cov = np.empty((n_obs, n_obs), dtype=float)
    row = 0
    for obs1, cols1 in groups:
        block_mean = np.asarray(pred_mean[obs1])
        mean_blocks.append(block_mean.reshape(1, -1)[:, : len(cols1)])
        col = 0
        for obs2, cols2 in groups:
            cov_block = np.asarray(pred_cov[obs1, obs2])
            cov[row : row + len(cols1), col : col + len(cols2)] = cov_block.reshape(1, len(cols1), len(cols2))[0]
            col += len(cols2)
        row += len(cols1)
    return np.concatenate(mean_blocks, axis=1).reshape(-1), cov


def _loglike_terms(residual: np.ndarray, cov: np.ndarray) -> dict[str, float]:
    try:
        chol = np.linalg.cholesky(cov)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
        chi2 = float(residual @ alpha)
        logdet = float(2.0 * np.sum(np.log(np.diag(chol))))
        n = residual.size
        loglike_full = float(-0.5 * (chi2 + logdet + n * np.log(2.0 * np.pi)))
        loglike_jetscape = float(-0.5 * chi2 - 0.5 * logdet)
        return {
            "finite": 1.0,
            "chi2": chi2,
            "logdet": logdet,
            "loglike_full": loglike_full,
            "loglike_jetscape_convention": loglike_jetscape,
        }
    except np.linalg.LinAlgError:
        return {
            "finite": 0.0,
            "chi2": float("nan"),
            "logdet": float("nan"),
            "loglike_full": float("-inf"),
            "loglike_jetscape_convention": float("-inf"),
        }


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _compare_arrays(name: str, left: np.ndarray, right: np.ndarray, rows: list[dict[str, Any]]) -> None:
    left = np.asarray(left)
    right = np.asarray(right)
    row = {
        "checkpoint": name,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "same_shape": left.shape == right.shape,
        "max_abs": None,
        "mean_abs": None,
        "left_nan_count": int(np.isnan(left).sum()) if np.issubdtype(left.dtype, np.number) else None,
        "right_nan_count": int(np.isnan(right).sum()) if np.issubdtype(right.dtype, np.number) else None,
        "allclose_rtol1e-12_atol1e-12": False,
        "allclose_rtol1e-8_atol1e-10": False,
    }
    if left.shape == right.shape:
        diff = np.abs(left - right)
        row["max_abs"] = float(np.nanmax(diff)) if diff.size else 0.0
        row["mean_abs"] = float(np.nanmean(diff)) if diff.size else 0.0
        row["allclose_rtol1e-12_atol1e-12"] = bool(np.allclose(left, right, rtol=1e-12, atol=1e-12))
        row["allclose_rtol1e-8_atol1e-10"] = bool(np.allclose(left, right, rtol=1e-8, atol=1e-10))
    rows.append(row)
    LOGGER.info("CHECK %-34s same_shape=%s max_abs=%s", name, row["same_shape"], row["max_abs"])


def _resolve_existing_path(path: str | Path, repo: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if not candidate.is_absolute():
        for root in [repo, *repo.parents]:
            rooted = root / candidate
            if rooted.exists():
                return rooted
    return candidate


def _checkpoint_thetas(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path, repo: Path) -> np.ndarray:
    mins = np.array([float(p["min"]) for p in cfg["parameters"]], dtype=float)
    maxs = np.array([float(p["max"]) for p in cfg["parameters"]], dtype=float)
    theta_rows = [(mins + maxs) / 2.0]

    rng = np.random.default_rng(args.seed)
    random_points = rng.uniform(mins, maxs, size=(args.random_points, mins.size))
    if args.enforce_monotonic_first_three:
        random_points[:, :3].sort(axis=1)
    theta_rows.extend(random_points)

    for chain_csv in args.chain_csv:
        csv_path = _resolve_existing_path(chain_csv, repo)
        if not csv_path.exists():
            LOGGER.warning("Skipping missing chain CSV for theta checkpoints: %s", csv_path)
            continue
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            LOGGER.warning("Skipping empty chain CSV for theta checkpoints: %s", csv_path)
            continue
        indices = np.linspace(0, len(df) - 1, num=min(args.chain_points_per_file, len(df)), dtype=int)
        theta_rows.extend(df.iloc[indices, : mins.size].to_numpy(dtype=float))
        LOGGER.info("Added %d theta checkpoints from %s", len(indices), csv_path)

    thetas = np.vstack(theta_rows)
    pd.DataFrame(thetas, columns=[p["name"] for p in cfg["parameters"]]).to_csv(out_dir / "theta_checkpoints.csv", index=False)
    return thetas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Au+Au 200 emulator equivalence checkpoints.")
    parser.add_argument("--hic-config", type=Path, default=DEFAULT_HIC_CONFIG)
    parser.add_argument("--jetscape-config", type=Path, default=DEFAULT_JET_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--random-points", type=int, default=8)
    parser.add_argument("--chain-points-per-file", type=int, default=5)
    parser.add_argument("--no-monotonic-first-three", action="store_false", dest="enforce_monotonic_first_three")
    parser.add_argument("--save-arrays", action="store_true", help="Save prediction/covariance arrays for each theta as compressed NPZ.")
    parser.add_argument(
        "--skip-pointwise",
        action="store_true",
        help=(
            "Skip emulator.predict/log-likelihood checkpoints. Useful when loading a legacy dill "
            "under a different scikit-learn version causes native crashes; config/PCA/GP evidence "
            "is still written."
        ),
    )
    parser.add_argument(
        "--chain-csv",
        action="append",
        default=[
            "results/hic-bayes/AllDetectorsAuAu200/emulator/results/run_jetscape_mcmc/AuAu200_hicbayes/chain.csv",
            "results/hic-bayes/AllDetectorsAuAu200/emulator/results/run_jetscape_mcmc/AuAu200_jetscape/chain.csv",
        ],
        help="Posterior CSV used only to choose fixed theta checkpoints. Can be repeated.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path.cwd()
    sys.path.insert(0, str(repo))
    hic_config = args.hic_config if args.hic_config.is_absolute() else repo / args.hic_config
    jet_config = args.jetscape_config if args.jetscape_config.is_absolute() else repo / args.jetscape_config
    if args.out_dir is None:
        args.out_dir = repo / "studies/AllDetectorsAuAu200/emulator/diagnostics/auau200_equivalence"
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "diagnostics.log", mode="w"),
        ],
        force=True,
    )
    LOGGER.info("Writing diagnostics to %s", out_dir)
    _ensure_results_env(repo)
    LOGGER.info("hic-bayes config: %s", hic_config)
    LOGGER.info("JETSCAPE config: %s", jet_config)

    hic_cfg = _load_yaml(hic_config)
    jet_cfg = _load_yaml(jet_config)
    hic_emulator, hic_emulator_path = _load_emulator(hic_cfg, hic_config)
    jet_emulator, jet_emulator_path = _load_emulator(jet_cfg, jet_config)

    summary: dict[str, Any] = {
        "hic_config": str(hic_config),
        "jetscape_config": str(jet_config),
        "hic_emulator": str(hic_emulator_path),
        "jetscape_emulator": str(jet_emulator_path),
    }

    config_summary = {
        "parameter_names_equal": [p["name"] for p in hic_cfg["parameters"]] == [p["name"] for p in jet_cfg["parameters"]],
        "parameter_bounds_equal": [
            (p["min"], p["max"]) for p in hic_cfg["parameters"]
        ] == [
            (p["min"], p["max"]) for p in jet_cfg["parameters"]
        ],
        "observable_columns_equal": hic_cfg.get("conversion", {}).get("observable_columns") == jet_cfg.get("conversion", {}).get("observable_columns"),
        "obs_order_equal": hic_cfg.get("conversion", {}).get("obs_order") == jet_cfg.get("conversion", {}).get("obs_order"),
        "hic_design_remove_idx": hic_cfg.get("conversion", {}).get("design_remove_idx", []),
        "jetscape_design_remove_idx": jet_cfg.get("conversion", {}).get("design_remove_idx", []),
        "hic_covariance": hic_cfg.get("experimental_covariance", {}),
        "jetscape_covariance": jet_cfg.get("experimental_covariance", {}),
        "hic_transform_multiplicities": hic_cfg.get("jetscape", {}).get("transform_multiplicities"),
        "jetscape_transform_multiplicities": jet_cfg.get("jetscape", {}).get("transform_multiplicities"),
    }
    summary["config"] = config_summary
    _write_json(out_dir / "config_summary.json", config_summary)
    LOGGER.info("CONFIG parameter_names_equal=%s parameter_bounds_equal=%s observable_columns_equal=%s",
                config_summary["parameter_names_equal"], config_summary["parameter_bounds_equal"],
                config_summary["observable_columns_equal"])
    LOGGER.info("CONFIG design_remove_idx hic=%s jetscape=%s",
                config_summary["hic_design_remove_idx"], config_summary["jetscape_design_remove_idx"])

    rows: list[dict[str, Any]] = []

    hic_design_raw = _read_table(hic_cfg, "design", kind="model", config_path=hic_config)
    jet_design_raw = _read_table(jet_cfg, "design", kind="model", config_path=jet_config)
    hic_sim_raw = _read_table(hic_cfg, "simulation", kind="model", config_path=hic_config)
    jet_sim_raw = _read_table(jet_cfg, "simulation", kind="model", config_path=jet_config)
    hic_columns = _columns(hic_cfg, hic_sim_raw)
    jet_columns = _columns(jet_cfg, jet_sim_raw)
    hic_design, hic_sim = _apply_removed_design_points(hic_cfg, hic_design_raw, hic_sim_raw.loc[:, hic_columns])
    jet_design, jet_sim = _apply_removed_design_points(jet_cfg, jet_design_raw, jet_sim_raw.loc[:, jet_columns])
    _compare_arrays("design_raw", hic_design_raw.to_numpy(float), jet_design_raw.to_numpy(float), rows)
    _compare_arrays("simulation_raw_selected_columns", hic_sim_raw.loc[:, hic_columns].to_numpy(float), jet_sim_raw.loc[:, jet_columns].to_numpy(float), rows)
    _compare_arrays("design_after_config_removal", hic_design.to_numpy(float), jet_design.to_numpy(float), rows)
    _compare_arrays("simulation_after_config_removal", hic_sim.to_numpy(float), jet_sim.to_numpy(float), rows)

    hic_prepared = _prepare_data(hic_cfg, hic_config)
    jet_prepared = _prepare_data(jet_cfg, jet_config)
    _compare_arrays("prepared_y_exp", hic_prepared.y_exp, jet_prepared.y_exp, rows)
    _compare_arrays("prepared_exp_cov", hic_prepared.exp_cov, jet_prepared.exp_cov, rows)
    summary["prepared_groups_equal"] = hic_prepared.groups == jet_prepared.groups
    summary["n_prepared_groups"] = len(hic_prepared.groups)
    summary["n_observables"] = int(hic_prepared.y_exp.size)
    LOGGER.info("PREPARED groups_equal=%s n_groups=%d n_obs=%d",
                summary["prepared_groups_equal"], summary["n_prepared_groups"], summary["n_observables"])

    for attr in [
        "scaler_mean",
        "scaler_scale",
        "pca_components",
        "pca_explained_variance",
        "pca_explained_variance_ratio",
        "transform_matrix",
        "cov_trunc",
    ]:
        _compare_arrays(f"pca_{attr}", _pca_attr(hic_emulator, attr), _pca_attr(jet_emulator, attr), rows)

    component_dot = np.sum(
        _pca_attr(hic_emulator, "pca_components")[:10] * _pca_attr(jet_emulator, "pca_components")[:10],
        axis=1,
    )
    pd.DataFrame({"pc": np.arange(len(component_dot)), "component_dot": component_dot, "abs_component_dot": np.abs(component_dot)}).to_csv(
        out_dir / "pca_component_dot_products.csv",
        index=False,
    )

    gp_rows = []
    for idx, (hic_gp, jet_gp) in enumerate(zip(getattr(hic_emulator, "gps"), getattr(jet_emulator, "gps"))):
        hic_kernel = str(getattr(hic_gp, "kernel_", getattr(hic_gp, "kernel", "")))
        jet_kernel = str(getattr(jet_gp, "kernel_", getattr(jet_gp, "kernel", "")))
        hic_n_features = getattr(hic_gp, "n_features_in_", None)
        jet_n_features = getattr(jet_gp, "n_features_in_", None)
        hic_length_scale = getattr(getattr(hic_gp, "kernel_", None), "k1", None)
        jet_length_scale = getattr(getattr(jet_gp, "kernel_", None), "k1", None)
        hic_length_scale = getattr(getattr(hic_length_scale, "k2", hic_length_scale), "length_scale", None)
        jet_length_scale = getattr(getattr(jet_length_scale, "k2", jet_length_scale), "length_scale", None)
        gp_rows.append(
            {
                "pc": idx,
                "kernel_string_equal": hic_kernel == jet_kernel,
                "hic_n_features_in": hic_n_features,
                "jetscape_n_features_in": jet_n_features,
                "feature_dimension_equal": hic_n_features == jet_n_features,
                "hic_length_scale_count": None if hic_length_scale is None else int(np.size(hic_length_scale)),
                "jetscape_length_scale_count": None if jet_length_scale is None else int(np.size(jet_length_scale)),
                "hic_kernel": hic_kernel,
                "jetscape_kernel": jet_kernel,
            }
        )
        LOGGER.info(
            "GP pc=%02d kernel_equal=%s feature_dim hic=%s jetscape=%s",
            idx,
            hic_kernel == jet_kernel,
            hic_n_features,
            jet_n_features,
        )
    pd.DataFrame(gp_rows).to_csv(out_dir / "gp_kernel_comparison.csv", index=False)

    observable_summary = {
        "hic_observables": _observable_names(hic_emulator),
        "jetscape_observables": _observable_names(jet_emulator),
        "observables_equal": _observable_names(hic_emulator) == _observable_names(jet_emulator),
    }
    _write_json(out_dir / "observable_summary.json", observable_summary)
    LOGGER.info("OBSERVABLES equal=%s count_hic=%d count_jetscape=%d",
                observable_summary["observables_equal"], len(observable_summary["hic_observables"]),
                len(observable_summary["jetscape_observables"]))

    if args.skip_pointwise:
        LOGGER.warning("Skipping pointwise prediction/log-likelihood checkpoints by request.")
        with (out_dir / "checkpoint_array_comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary["pointwise_skipped"] = True
        _write_json(out_dir / "summary.json", summary)
        LOGGER.info("Done. Evidence files written under %s", out_dir)
        return

    thetas = _checkpoint_thetas(args, hic_cfg, out_dir, repo)
    point_rows = []
    for idx, theta in enumerate(thetas):
        hic_mean, hic_cov_emu = _flatten_prediction(hic_emulator, hic_prepared.groups, theta)
        jet_mean, jet_cov_emu = _flatten_prediction(jet_emulator, jet_prepared.groups, theta)
        hic_total_cov = hic_cov_emu + hic_prepared.exp_cov
        jet_total_cov = jet_cov_emu + jet_prepared.exp_cov
        hic_residual = hic_mean - hic_prepared.y_exp
        jet_residual = jet_mean - jet_prepared.y_exp
        hic_terms = _loglike_terms(hic_residual, hic_total_cov)
        jet_terms = _loglike_terms(jet_residual, jet_total_cov)
        row = {
            "theta_index": idx,
            "prediction_mean_max_abs": _max_abs(hic_mean, jet_mean),
            "prediction_mean_mean_abs": _mean_abs(hic_mean, jet_mean),
            "emu_cov_max_abs": _max_abs(hic_cov_emu, jet_cov_emu),
            "emu_cov_mean_abs": _mean_abs(hic_cov_emu, jet_cov_emu),
            "total_cov_max_abs": _max_abs(hic_total_cov, jet_total_cov),
            "residual_max_abs": _max_abs(hic_residual, jet_residual),
            "hic_chi2": hic_terms["chi2"],
            "jetscape_chi2": jet_terms["chi2"],
            "delta_chi2_hic_minus_jetscape": hic_terms["chi2"] - jet_terms["chi2"],
            "hic_logdet": hic_terms["logdet"],
            "jetscape_logdet": jet_terms["logdet"],
            "delta_logdet_hic_minus_jetscape": hic_terms["logdet"] - jet_terms["logdet"],
            "hic_loglike_full": hic_terms["loglike_full"],
            "jetscape_loglike_full": jet_terms["loglike_full"],
            "delta_loglike_full_hic_minus_jetscape": hic_terms["loglike_full"] - jet_terms["loglike_full"],
            "hic_loglike_jetscape_convention": hic_terms["loglike_jetscape_convention"],
            "jetscape_loglike_jetscape_convention": jet_terms["loglike_jetscape_convention"],
            "delta_loglike_jetscape_convention_hic_minus_jetscape": hic_terms["loglike_jetscape_convention"] - jet_terms["loglike_jetscape_convention"],
        }
        point_rows.append(row)
        LOGGER.info(
            "THETA %03d mean_max=% .6e cov_max=% .6e dchi2=% .6e dlogdet=% .6e dloglike=% .6e",
            idx,
            row["prediction_mean_max_abs"],
            row["total_cov_max_abs"],
            row["delta_chi2_hic_minus_jetscape"],
            row["delta_logdet_hic_minus_jetscape"],
            row["delta_loglike_jetscape_convention_hic_minus_jetscape"],
        )
        if args.save_arrays:
            np.savez_compressed(
                out_dir / f"theta_{idx:03d}_arrays.npz",
                theta=theta,
                hic_mean=hic_mean,
                jetscape_mean=jet_mean,
                hic_emu_cov=hic_cov_emu,
                jetscape_emu_cov=jet_cov_emu,
                hic_total_cov=hic_total_cov,
                jetscape_total_cov=jet_total_cov,
                hic_residual=hic_residual,
                jetscape_residual=jet_residual,
            )

    with (out_dir / "checkpoint_array_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(point_rows).to_csv(out_dir / "pointwise_likelihood_comparison.csv", index=False)
    summary["max_delta_loglike_jetscape_convention"] = float(
        np.nanmax(np.abs([r["delta_loglike_jetscape_convention_hic_minus_jetscape"] for r in point_rows]))
    )
    summary["max_prediction_mean_abs"] = float(np.nanmax([r["prediction_mean_max_abs"] for r in point_rows]))
    summary["max_total_cov_abs"] = float(np.nanmax([r["total_cov_max_abs"] for r in point_rows]))
    _write_json(out_dir / "summary.json", summary)
    LOGGER.info("Done. Evidence files written under %s", out_dir)


if __name__ == "__main__":
    main()
