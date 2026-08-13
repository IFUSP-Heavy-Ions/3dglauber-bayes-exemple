"""Run an LHC prediction MCMC with a pre-trained JETSCAPE emulator.

This pipeline is intentionally separate from ``run_universality_analysis.py``.
It mirrors the MCMC/likelihood conventions in the JETSCAPE 3D calibration code
(``bayes_mcmc.py`` and ``emulator.py``) while reading hic-bayes style YAMLs and
flat CSV data tables.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import pickle
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import emcee
import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.linalg import lapack

from hic_bayes.emulator_jet import JetCompatibleEmulator, transform_jet_design
from hic_bayes.io import read_table, table_path

LOGGER = logging.getLogger(__name__)

try:
    import dill
except ImportError:  # pragma: no cover - only triggered in an incomplete env
    dill = None


def _load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def _analysis_dir(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "sims" else config_path.parent


def _resolve_path(base: Path, path: str) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    unresolved = re.search(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*", expanded)
    if unresolved:
        raise ValueError(f"Unresolved environment variable in path: {path!r}")
    candidate = Path(expanded)
    return candidate if candidate.is_absolute() else base / candidate


def _resolve_output_file(config_path: Path, mcmc_cfg: dict) -> Path:
    base = _analysis_dir(config_path)
    env_results_dir = os.environ.get("HIC_BAYES_RESULTS_DIR")
    run_tag = str(mcmc_cfg.get("run_tag", "run"))
    if env_results_dir:
        path = Path(os.path.expanduser(os.path.expandvars(env_results_dir))) / run_tag / "chain.hdf5"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    output_root = mcmc_cfg.get("output_root", "results")
    root = _resolve_path(base, output_root)
    if mcmc_cfg.get("shared_run_dir", False):
        run_id = mcmc_cfg.get("run_id") or os.environ.get("HIC_BAYES_RUN_ID")
        if run_id:
            path = root.parent / str(run_id) / root.name / run_tag / "chain.hdf5"
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
    path = root / run_tag / "chain.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_emulator_file(config_path: Path, cfg: dict) -> Path:
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


def _load_emulator(path: Path, source_root: str | None = None):
    source_path: Path | None = None
    if source_root:
        source_path = Path(os.path.expanduser(os.path.expandvars(str(source_root))))
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        source_path = source_path.resolve()
        LOGGER.info("Using JETSCAPE source root: %s", source_path)
        sys.path.insert(0, str(source_path))
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        if path.suffix == ".dill":
            if dill is None:
                raise RuntimeError("Loading .dill emulators requires the 'dill' package.")
            _prepare_legacy_dill_namespace(source_path)
            emulator = dill.load(handle)
            _patch_legacy_emulator_globals(emulator, source_path=source_path)
            _log_loaded_emulator(emulator)
            return emulator
        try:
            emulator = pickle.load(handle)
        except Exception:
            if dill is None:
                raise
            handle.seek(0)
            _prepare_legacy_dill_namespace(source_path)
            emulator = dill.load(handle)
            _patch_legacy_emulator_globals(emulator, source_path=source_path)
        _log_loaded_emulator(emulator)
        return emulator


def _import_legacy_module(name: str, source_path: Path | None):
    """Import a module from jetscape.source_root, avoiding stale name clashes."""
    module = sys.modules.get(name)
    if source_path is not None and module is not None:
        module_file = getattr(module, "__file__", None)
        try:
            module_path = Path(module_file).resolve() if module_file else None
        except OSError:
            module_path = None
        if module_path is None or source_path not in [module_path, *module_path.parents]:
            LOGGER.info("Reloading legacy module %s from %s", name, source_path)
            del sys.modules[name]
    return importlib.import_module(name)


@contextmanager
def _temporary_cwd(path: Path | None):
    if path is None:
        yield
        return
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _legacy_dill_globals(source_path: Path | None) -> dict[str, object]:
    try:
        with _temporary_cwd(source_path):
            legacy_configurations = _import_legacy_module("configurations", source_path)
            legacy_emulator = _import_legacy_module("emulator", source_path)
    except ImportError:
        if source_path:
            LOGGER.warning("Could not import legacy JETSCAPE modules from %s.", source_path)
        return {}

    globals_map: dict[str, object] = {
        "_configurations": legacy_configurations,
        "do_transform_design": legacy_configurations.do_transform_design,
        "transform_design": legacy_configurations.transform_design,
    }
    if hasattr(legacy_emulator, "_Covariance"):
        globals_map["_Covariance"] = legacy_emulator._Covariance
    return globals_map


def _prepare_legacy_dill_namespace(source_path: Path | None) -> None:
    """Populate __main__ for JETSCAPE dill objects saved as __main__.Emulator."""
    globals_map = _legacy_dill_globals(source_path)
    if not globals_map:
        return
    import __main__ as main_module

    for name, value in globals_map.items():
        setattr(main_module, name, value)
    LOGGER.info("Prepared legacy JETSCAPE dill namespace with: %s", sorted(globals_map))


def _patch_legacy_emulator_globals(emulator, source_path: Path | None = None) -> None:
    """Provide globals expected by JETSCAPE dill-bound emulator methods."""
    predict = getattr(emulator, "predict", None)
    predict_globals = getattr(predict, "__globals__", None)
    if predict_globals is None:
        return
    globals_map = _legacy_dill_globals(source_path)
    if not globals_map:
        return
    predict_globals.update(globals_map)


def _external_dill_blocked_message(emulator_path: Path) -> str:
    return (
        "Refusing to load an external JETSCAPE .dill emulator by default: "
        f"{emulator_path}. dill can deserialize objects bound to classes in "
        "jetscape-emulator/emulator.py, so using it here makes this simulation "
        "depend on code outside hic-bayes. Retrain/cache a hic-bayes emulator "
        "instead by setting emulator.train_in_hic_bayes: true and using a .pkl "
        "cache_file. To intentionally run the legacy external object, set "
        "emulator.allow_external_dill: true."
    )


def _log_loaded_emulator(emulator) -> None:
    module_name = type(emulator).__module__
    module = sys.modules.get(module_name)
    LOGGER.info(
        "Loaded emulator object: type=%s.%s module_file=%s",
        module_name,
        type(emulator).__name__,
        getattr(module, "__file__", None),
    )
    legacy_module = sys.modules.get("emulator")
    if legacy_module is not None:
        LOGGER.info("Active legacy emulator module: %s", getattr(legacy_module, "__file__", None))


def _export_chain_csv(chain_file: Path, param_names: list[str], thin: int = 1) -> Path:
    """Write a flat posterior CSV next to the HDF5 chain for plotting."""
    thin = max(1, int(thin))
    with h5py.File(chain_file, "r") as handle:
        chain = handle["chain"][:]

    if chain.ndim != 3:
        raise ValueError(f"Expected chain dataset with 3 dimensions, got shape {chain.shape}.")
    if chain.shape[1] == 0:
        raise ValueError(f"Cannot export an empty chain dataset: {chain_file}")
    if chain.shape[2] != len(param_names):
        raise ValueError(
            f"Chain dimension mismatch for {chain_file}: "
            f"{chain.shape[2]} parameters in chain, {len(param_names)} names in YAML."
        )

    flat_chain = chain[:, ::thin, :].reshape(-1, chain.shape[-1])
    csv_path = chain_file.with_suffix(".csv")
    pd.DataFrame(flat_chain, columns=param_names).to_csv(csv_path, index=False)
    LOGGER.info("Saved posterior chain CSV to: %s", csv_path)
    return csv_path


def _load_or_train_emulator(config_path: Path, cfg: dict):
    em_cfg = cfg.get("emulator", {})
    train_in_hic_bayes = bool(
        em_cfg.get("train", False)
        or em_cfg.get("train_in_hic_bayes", False)
        or str(em_cfg.get("source", "")).strip().lower() == "hic-bayes"
    )
    emulator_path = _resolve_emulator_file(config_path, cfg)

    if not train_in_hic_bayes:
        if emulator_path.suffix == ".dill" and not bool(em_cfg.get("allow_external_dill", False)):
            raise RuntimeError(_external_dill_blocked_message(emulator_path))
        source_root = cfg.get("jetscape", {}).get("source_root")
        if source_root and not bool(em_cfg.get("allow_external_dill", False)):
            raise RuntimeError(
                "Refusing to add jetscape.source_root to sys.path by default. "
                "This would couple the run to code outside hic-bayes. "
                "Set emulator.train_in_hic_bayes: true and use a .pkl cache_file, "
                "or set emulator.allow_external_dill: true for an intentional legacy run."
            )
        LOGGER.info("Loading JETSCAPE emulator: %s", emulator_path)
        return _load_emulator(emulator_path, source_root=source_root)

    transform_design = bool(cfg.get("jetscape", {}).get("transform_design", True))
    if emulator_path.exists() and not bool(em_cfg.get("retrain", False)):
        LOGGER.info("Loading hic-bayes trained JETSCAPE-compatible emulator: %s", emulator_path)
        emulator = _load_emulator(emulator_path)
        if _cached_hicbayes_emulator_is_compatible(emulator, cfg, transform_design):
            return emulator
        LOGGER.warning("Cached hic-bayes emulator is incompatible with current JETSCAPE design transform; retraining.")

    LOGGER.info("Training hic-bayes JETSCAPE-compatible emulator from flat CSV tables.")
    design_df = read_table(cfg, "design", kind="model")
    sim_df = read_table(cfg, "simulation", kind="model")
    observable_columns = cfg.get("conversion", {}).get("observable_columns") or list(sim_df.columns)
    sim_df = sim_df.loc[:, observable_columns]
    system_name = str(cfg.get("conversion", {}).get("source_system") or cfg.get("system", "<unknown>"))
    idf = cfg.get("jetscape", {}).get("idf", cfg.get("emulator", {}).get("idf", 0))
    idf_label = cfg.get("jetscape", {}).get("idf_label", "Grad" if str(idf) == "0" else "")
    _log_training_inputs(cfg, system_name, idf, idf_label, design_df, sim_df, observable_columns)

    removed_design_idx = [int(x) for x in cfg.get("conversion", {}).get("design_remove_idx", [])]
    if removed_design_idx:
        keep_mask = np.ones(len(design_df), dtype=bool)
        keep_mask[removed_design_idx] = False
        design_df = design_df.loc[keep_mask].copy()
        sim_df = sim_df.loc[keep_mask].copy()
        LOGGER.info("Removed %d design points before training: %s", len(removed_design_idx), removed_design_idx)

    design = design_df.to_numpy(dtype=float)
    simulation = sim_df.to_numpy(dtype=float)
    if not np.isfinite(simulation).all():
        sim_df = _impute_simulation_blocks(sim_df)
        simulation = sim_df.to_numpy(dtype=float)
        if not np.isfinite(simulation).all():
            raise ValueError("simulation.csv contains non-finite values that could not be imputed.")
        LOGGER.warning("Imputed non-finite simulation.csv values within observable blocks before training.")

    param_ranges = np.array(
        [[float(param["min"]), float(param["max"])] for param in cfg["parameters"]],
        dtype=float,
    )
    n_components = int(cfg.get("pca", {}).get("n_components", em_cfg.get("n_components", 10)))
    n_restarts = int(em_cfg.get("n_restarts", 4))
    n_jobs = int(os.environ.get("HIC_BAYES_NTHREADS", em_cfg.get("n_jobs", -1)))
    LOGGER.info(
        "Training settings: design=%s, simulation=%s, npc=%d, n_restarts=%d, n_jobs=%d",
        design.shape,
        simulation.shape,
        n_components,
        n_restarts,
        n_jobs,
    )
    emulator = JetCompatibleEmulator.fit(
        design=design,
        simulation=simulation,
        param_ranges=param_ranges,
        observable_columns=observable_columns,
        n_components=n_components,
        n_restarts=n_restarts,
        n_jobs=n_jobs,
        system_name=system_name,
        idf=idf,
        transform_design=transform_design,
    )

    emulator_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = emulator_path.with_name(f".{emulator_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(emulator, handle)
    os.replace(tmp_path, emulator_path)
    LOGGER.info("Saved hic-bayes trained emulator to: %s", emulator_path)
    return emulator


def _cached_hicbayes_emulator_is_compatible(emulator, cfg: dict, transform_design: bool) -> bool:
    if not isinstance(emulator, JetCompatibleEmulator):
        return True
    gps = getattr(emulator, "gps", [])
    if not gps:
        LOGGER.warning("Cached hic-bayes emulator has no GP list.")
        return False
    n_features = getattr(gps[0], "n_features_in_", None)
    n_params = len(cfg["parameters"])
    expected_features = transform_jet_design(np.zeros((1, n_params))).shape[1] if transform_design else n_params
    cached_transform = bool(getattr(emulator, "transform_design", False))
    if cached_transform != transform_design:
        LOGGER.warning(
            "Cached hic-bayes emulator transform_design=%s but config requests %s.",
            cached_transform,
            transform_design,
        )
        return False
    if n_features is not None and int(n_features) != expected_features:
        LOGGER.warning(
            "Cached hic-bayes emulator GP feature dimension is %s; expected %s.",
            n_features,
            expected_features,
        )
        return False
    return True


def _observable_names(columns: list[str]) -> list[str]:
    names: list[str] = []
    seen = set()
    for column in columns:
        obs = column.rsplit("__bin", 1)[0] if "__bin" in column else column
        if obs not in seen:
            seen.add(obs)
            names.append(obs)
    return names


def _impute_simulation_blocks(sim_df: pd.DataFrame) -> pd.DataFrame:
    out = sim_df.copy()
    col_groups: dict[str, list[str]] = {}
    for col in out.columns:
        key = str(col).split("__bin", 1)[0]
        col_groups.setdefault(key, []).append(col)

    global_means = out.replace([np.inf, -np.inf], np.nan).mean(axis=0)
    for cols in col_groups.values():
        block = out[cols].replace([np.inf, -np.inf], np.nan)
        for row_idx in block.index:
            row = block.loc[row_idx]
            if row.isna().any():
                fill_value = row.dropna().mean()
                if np.isnan(fill_value):
                    fill_value = global_means[cols].mean()
                out.loc[row_idx, cols] = row.fillna(fill_value)

    return out


def _log_nonfinite_model_values(sim_df, observable_columns: list[str]) -> None:
    values = sim_df.to_numpy(dtype=float)
    if np.isfinite(values).all():
        LOGGER.info("No non-finite values found in model simulation table.")
        return
    LOGGER.warning("Found non-finite values in model simulation table.")
    for row, col in np.argwhere(~np.isfinite(values)):
        column = observable_columns[col]
        obs = column.rsplit("__bin", 1)[0] if "__bin" in column else column
        LOGGER.warning("Non-finite model data: design point %d, observable %s, column %s", row, obs, column)


def _log_training_inputs(cfg: dict, system_name: str, idf, idf_label: str, design_df, sim_df, observable_columns: list[str]) -> None:
    LOGGER.info("========== hic-bayes emulator training context ==========")
    LOGGER.info("Using idf = %s%s", idf, f": {idf_label}" if idf_label else "")
    active = {system_name: _observable_names(observable_columns)}
    LOGGER.info("The active observable list for calibration: %s", active)
    LOGGER.info("Loading %s main design from %s", system_name, table_path(cfg, "design", kind="model"))
    LOGGER.info("Design table shape = %s", design_df.shape)
    LOGGER.info("Loading %s main calculations from %s", system_name, table_path(cfg, "simulation", kind="model"))
    LOGGER.info("model_data.shape = %s", sim_df.shape)
    _log_nonfinite_model_values(sim_df, observable_columns)
    removed = cfg.get("conversion", {}).get("design_remove_idx", [])
    if removed:
        LOGGER.info("Design points recorded for deletion in source metadata: %s", removed)
    else:
        LOGGER.info("No design points recorded for deletion in source metadata.")
    try:
        val_design_df = read_table(cfg, "validation_design", kind="model")
        val_sim_df = read_table(cfg, "validation_simulation", kind="model")
    except (KeyError, FileNotFoundError):
        LOGGER.info("No validation design/simulation tables configured for %s.", system_name)
    else:
        LOGGER.info("Loading %s validation design from %s", system_name, table_path(cfg, "validation_design", kind="model"))
        LOGGER.info("validation_design.shape = %s", val_design_df.shape)
        LOGGER.info("Loading %s validation calculations from %s", system_name, table_path(cfg, "validation_simulation", kind="model"))
        LOGGER.info("validation_data.shape = %s", val_sim_df.shape)
    map_path = cfg.get("model", {}).get("map") or cfg.get("data", {}).get("map")
    LOGGER.info("Loading %s MAP calculations from %s", system_name, map_path)
    if not map_path:
        LOGGER.info("No MAP calculations found for system %s", system_name)
    LOGGER.info("=========================================================")


def jetscape_mvn_loglike(y: np.ndarray, cov: np.ndarray) -> float:
    """Non-normalized multivariate-normal log-likelihood from JETSCAPE."""
    L, info = lapack.dpotrf(cov, clean=False)
    if info != 0:
        return -np.inf
    alpha, info = lapack.dpotrs(L, y)
    if info != 0:
        return -np.inf
    np.fill_diagonal(L, np.maximum(1e-10, np.diag(L)))
    return float(-0.5 * np.dot(y, alpha) - np.log(L.diagonal()).sum())


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


def prepare_data(cfg: dict) -> PreparedData:
    exp_df = read_table(cfg, "experiment", kind="data")
    err_df = read_table(cfg, "exp_error", kind="data")
    columns = cfg.get("conversion", {}).get("observable_columns") or list(exp_df.columns)
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


class JetscapeChain:
    def __init__(
        self,
        cfg: dict,
        emulator,
        prepared: PreparedData,
        out_file: Path,
        emulator_path: Path | None = None,
        source_root: str | None = None,
    ):
        self.cfg = cfg
        self.emulator = emulator
        self.prepared = prepared
        self.out_file = out_file
        self.emulator_path = emulator_path
        self.source_root = source_root
        params = cfg["parameters"]
        self.names = [p["name"] for p in params]
        self.min = np.array([float(p["min"]) for p in params])
        self.max = np.array([float(p["max"]) for p in params])
        self.ndim = len(params)
        self.prior_volume = float(np.prod(self.max - self.min))
        self.monotonic_first_three = bool(cfg.get("jetscape", {}).get("monotonic_first_three", True))

    def random_pos(self, n: int, shape_prefix: tuple[int, ...] = ()) -> np.ndarray:
        return np.random.uniform(self.min, self.max, shape_prefix + (n, self.ndim))

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        X = np.array(theta, copy=False, ndmin=2)
        lp = np.log(np.ones(X.shape[0]) / self.prior_volume)
        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf
        if self.monotonic_first_three:
            mono = (X[:, 0] <= X[:, 1]) & (X[:, 1] <= X[:, 2])
            lp[~mono] = -np.inf
        return lp

    def _predict_flat(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred_mean, pred_cov = self.emulator.predict(X, return_cov=True)
        mean_blocks = []
        n_samples = X.shape[0]
        n_obs = self.prepared.y_exp.size
        cov = np.empty((n_samples, n_obs, n_obs))
        row = 0
        for obs1, cols1 in self.prepared.groups:
            if obs1 not in pred_mean:
                raise KeyError(
                    f"Emulator prediction is missing observable {obs1!r}. "
                    "Use matching observable names in simulation/experiment columns."
                )
            block_mean = np.asarray(pred_mean[obs1])
            if block_mean.shape[-1] != len(cols1):
                raise ValueError(
                    f"Observable {obs1!r} has {len(cols1)} data bins but emulator returns "
                    f"{block_mean.shape[-1]} bins."
                )
            mean_blocks.append(block_mean)
            col = 0
            for obs2, cols2 in self.prepared.groups:
                cov[:, row : row + len(cols1), col : col + len(cols2)] = pred_cov[obs1, obs2]
                col += len(cols2)
            row += len(cols1)
        return np.concatenate(mean_blocks, axis=1), cov

    def log_posterior(self, theta: np.ndarray) -> np.ndarray | float:
        scalar_input = np.asarray(theta).ndim == 1
        X = np.array(theta, copy=False, ndmin=2)
        lp = np.zeros(X.shape[0])
        inside = np.all((X > self.min) & (X < self.max), axis=1)
        if self.monotonic_first_three:
            inside &= (X[:, 0] <= X[:, 1]) & (X[:, 1] <= X[:, 2])
        lp[~inside] = -np.inf
        if np.any(inside):
            mean, cov = self._predict_flat(X[inside])
            cov += self.prepared.exp_cov
            dY = mean - self.prepared.y_exp
            lp[inside] += [jetscape_mvn_loglike(dy, c) for dy, c in zip(dY, cov)]
        return float(lp[0]) if scalar_input else lp

    def log_likelihood(self, theta: np.ndarray) -> np.ndarray | float:
        return self.log_posterior(theta)

    def run_emcee(self, nsteps: int, nburnsteps: int, nwalkers: int, status: int | None, ncpu: int) -> None:
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.out_file, "a") as handle:
            if "chain" in handle:
                dset = handle["chain"]
                nwalkers = dset.shape[0]
                burn = False
                if dset.shape[1] == 0:
                    print("Existing chain dataset is empty; running initial burn-in.", flush=True)
                    burn = True
            else:
                burn = True
                dset = handle.create_dataset(
                    "chain",
                    dtype="f8",
                    shape=(nwalkers, 0, self.ndim),
                    chunks=(nwalkers, 1, self.ndim),
                    maxshape=(nwalkers, None, self.ndim),
                    compression="lzf",
                )

            use_pool = bool(ncpu and ncpu > 1)
            if use_pool and self.emulator_path is None:
                LOGGER.warning("No emulator path available for worker initialization; running emcee without multiprocessing.")
                use_pool = False
            if use_pool:
                pool_context = Pool(
                    processes=ncpu,
                    initializer=_init_jetscape_worker,
                    initargs=(
                        self.cfg,
                        self.prepared,
                        str(self.out_file),
                        str(self.emulator_path),
                        self.source_root,
                    ),
                )
                log_prob_fn = _worker_log_posterior
                LOGGER.info("Using multiprocessing Pool with %d initialized JETSCAPE workers for emcee.", ncpu)
            else:
                pool_context = nullcontext(None)
                log_prob_fn = self.log_posterior
                LOGGER.info("Running emcee without multiprocessing.")

            with pool_context as pool:
                sampler = emcee.EnsembleSampler(nwalkers, self.ndim, log_prob_fn, pool=pool)
                if burn:
                    print("No existing chain found; running JETSCAPE-style two-stage burn-in.", flush=True)
                    nburn0 = nburnsteps // 2
                    self._run_with_status(sampler, self.random_pos(nwalkers), nburn0, status)
                    flat = sampler.get_chain(flat=True)
                    flat_logp = sampler.get_log_prob(flat=True)
                    best = np.unique(flat_logp, return_index=True)[1][-nwalkers:]
                    x0 = flat[best]
                    state = self._run_with_status(sampler, x0, nburnsteps - nburn0, status, store=False)
                    state = state.coords
                    sampler.reset()
                else:
                    print("Restarting from last point of existing chain.", flush=True)
                    state = dset[:, -1, :]

                print(f"Running emcee production: {nwalkers} walkers x {nsteps} steps.", flush=True)
                self._run_with_status(sampler, state, nsteps, status)
                dset.resize(dset.shape[1] + nsteps, 1)
                dset[:, -nsteps:, :] = sampler.get_chain().transpose(1, 0, 2)
                print(f"Wrote chain to {self.out_file}", flush=True)

    @staticmethod
    def _run_with_status(sampler, initial_state, nsteps: int, status: int | None, store: bool = True):
        status = status or max(1, nsteps // 10)
        t0 = time.time()
        state = None
        for step, state in enumerate(
            sampler.sample(initial_state, iterations=nsteps, store=store),
            start=1,
        ):
            if step % status == 0 or step == nsteps:
                af = sampler.acceptance_fraction
                print(
                    f"step {step}: acceptance mean={af.mean():.4f}, "
                    f"std={af.std():.4f}, min={af.min():.4f}, max={af.max():.4f}",
                    flush=True,
                )
        print(f"Finished {nsteps} steps in {time.time() - t0:.1f} s", flush=True)
        return state

    def run_ptemcee(self, nsteps: int, nburnsteps: int, nwalkers: int, ntemps: int, ncpu: int) -> None:
        import ptemcee

        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.out_file, "a") as handle:
            if "chain" in handle:
                raise RuntimeError("ptemcee resume is not implemented for this compatibility pipeline.")
            dset = handle.create_dataset(
                "chain",
                dtype="f8",
                shape=(nwalkers, 0, self.ndim),
                chunks=(nwalkers, 1, self.ndim),
                maxshape=(nwalkers, None, self.ndim),
                compression="lzf",
            )
            with Pool(ncpu) as pool:
                sampler = ptemcee.Sampler(
                    nwalkers,
                    self.ndim,
                    lambda x: float(self.log_likelihood(x)),
                    lambda x: float(self.log_prior(x)[0]),
                    ntemps,
                    np.inf,
                    pool=pool,
                )
                pos0 = self.random_pos(nwalkers, shape_prefix=(ntemps,))
                sampler.run_mcmc(pos0, nburnsteps, adapt=True)
                pos0 = sampler.chain[:, :, -1, :]
                sampler.reset()
                sampler.run_mcmc(pos0, nsteps)
            dset.resize(dset.shape[1] + nsteps, 1)
            dset[:, -nsteps:, :] = sampler.chain[0, :, :, :]
            print(f"Wrote zero-temperature ptemcee chain to {self.out_file}")


_WORKER_CHAIN: JetscapeChain | None = None


def _init_jetscape_worker(
    cfg: dict,
    prepared: PreparedData,
    out_file: str,
    emulator_path: str,
    source_root: str | None,
) -> None:
    """Load the emulator inside a worker so Pool does not pickle it."""
    global _WORKER_CHAIN
    emulator = _load_emulator(Path(emulator_path), source_root=source_root)
    _WORKER_CHAIN = JetscapeChain(
        cfg,
        emulator,
        prepared,
        Path(out_file),
        emulator_path=Path(emulator_path),
        source_root=source_root,
    )


def _worker_log_posterior(theta: np.ndarray) -> np.ndarray | float:
    if _WORKER_CHAIN is None:
        raise RuntimeError("JETSCAPE worker was not initialized.")
    return _WORKER_CHAIN.log_posterior(theta)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description="JETSCAPE-compatible LHC MCMC pipeline")
    parser.add_argument("config", help="Path to study YAML.")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = _load_config(config_path)
    emulator_path = _resolve_emulator_file(config_path, cfg)
    emulator = _load_or_train_emulator(config_path, cfg)
    prepared = prepare_data(cfg)
    out_file = _resolve_output_file(config_path, cfg["mcmc"])
    chain = JetscapeChain(
        cfg,
        emulator,
        prepared,
        out_file,
        emulator_path=emulator_path,
        source_root=cfg.get("jetscape", {}).get("source_root"),
    )

    mcmc = cfg["mcmc"]
    seed = mcmc.get("seed")
    if seed is not None:
        np.random.seed(int(seed))
        LOGGER.info("Set NumPy random seed for MCMC: %s", seed)

    sampler = str(mcmc.get("sampler", "emcee")).lower()
    if sampler == "ptemcee":
        chain.run_ptemcee(
            nsteps=int(mcmc["n_steps"]),
            nburnsteps=int(mcmc.get("burnin", 500)),
            nwalkers=int(mcmc["n_walkers"]),
            ntemps=int(mcmc.get("n_temps", 10)),
            ncpu=int(os.environ.get("HIC_BAYES_NTHREADS", mcmc.get("n_threads", cpu_count()))),
        )
    elif sampler == "emcee":
        chain.run_emcee(
            nsteps=int(mcmc["n_steps"]),
            nburnsteps=int(mcmc.get("burnin", 500)),
            nwalkers=int(mcmc["n_walkers"]),
            status=mcmc.get("status"),
            ncpu=int(os.environ.get("HIC_BAYES_NTHREADS", mcmc.get("n_threads", cpu_count()))),
        )
    else:
        raise ValueError(f"Unknown sampler {sampler!r}; use 'emcee' or 'ptemcee'.")

    _export_chain_csv(
        out_file,
        [str(param["name"]) for param in cfg["parameters"]],
        thin=int(mcmc.get("thin", 1)),
    )


if __name__ == "__main__":
    main()
