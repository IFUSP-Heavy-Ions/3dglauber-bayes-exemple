"""Bayesian parameter estimation for the Universality scaled-spectra analysis.

Faithful reproduction of ``Bayesian_pipeline.py`` (Universality_Github) inside
the hic-bayes framework.  Key design choices that match the original:

* SVD-based PCA with whitened PC scores (not sklearn PCA).
* Likelihood evaluated in **observable space** (287-dim), not PC space.
* Task-local sklearn GP emulators cached under ``<task_dir>/emulator``.
  Emulator pickles are analysis products and are not read from or written to
  ``data/``.
* emcee MCMC by default, with optional ptemcee parallel-tempering support.
* Supports diagonal *and* equicorrelated experimental covariance via the config.

Usage
-----
From the hic-bayes root directory:

    python -m pipelines.run_universality_analysis \\
        config/universality_PbPb2760_Grad_diag.yaml

    python -m pipelines.run_universality_analysis \\
        config/universality_PbPb2760_Grad_equicorr_rho020.yaml

    # Force emulator retraining:
    python -m pipelines.run_universality_analysis config/... --retrain-emulators
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import subprocess
import tempfile
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import StandardScaler

from hic_bayes.io import read_table
from hic_bayes.likelihood import (
    build_experimental_covariance,
    build_total_covariance_obs,
    mvn_loglike,
)
from hic_bayes.mcmc import load_chain_hdf5, run_emcee, run_ptemcee, thin_chain
from hic_bayes.plots import plot_trace
from hic_bayes.posterior import corner_plot, map_estimate, posterior_summary
from hic_bayes.prior import UniformPrior


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _output_base_dir(config_path: Path) -> Path:
    """Return the analysis directory that owns a task YAML."""
    config_dir = config_path.parent
    return config_dir.parent if config_dir.name == "tasks" else config_dir


def _resolve_output_path(output_base: Path, path: str) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    unresolved_env = re.search(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*", expanded)
    if unresolved_env:
        raise ValueError(
            f"Unresolved environment variable in path: {path!r}. "
            "Define the variable or remove it from the YAML."
        )

    out_path = Path(expanded)
    if out_path.is_absolute():
        return out_path
    return output_base / out_path


def _mkdir_existing_ok(path: Path) -> None:
    """Create a directory, tolerating concurrent creation on shared filesystems."""
    for _ in range(10):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except (FileExistsError, FileNotFoundError):
            if path.is_dir():
                return
            time.sleep(0.1)

    path.mkdir(parents=True, exist_ok=True)


def _resolve_output_file(mcmc_cfg: dict, output_base: Path) -> Path:
    """Resolve the output path, allocating a numbered run folder if requested."""
    env_results_dir = os.environ.get("HIC_BAYES_RESULTS_DIR")
    output_root = mcmc_cfg.get("output_root")

    if env_results_dir:
        run_tag = str(mcmc_cfg.get("run_tag", "run"))
        output_group = Path(str(output_root)).name if output_root else "results"
        root = Path(os.path.expanduser(os.path.expandvars(env_results_dir))) / output_group
        tag_dir = root / run_tag
        _mkdir_existing_ok(tag_dir)
        return tag_dir / "chain.hdf5"

    if output_root:
        root = _resolve_output_path(output_base, output_root)
        run_tag = str(mcmc_cfg.get("run_tag", "run"))
        if mcmc_cfg.get("shared_run_dir", False):
            run_parent = root.parent
            output_group = root.name
            run_id = mcmc_cfg.get("run_id") or os.environ.get("HIC_BAYES_RUN_ID")

            if run_id:
                run_id = _slug(str(run_id))
                tag_dir = run_parent / run_id / output_group / run_tag
                _mkdir_existing_ok(tag_dir)
                return tag_dir / "chain.hdf5"

            run_index = 1
            while True:
                tag_dir = run_parent / f"run_{run_index:03d}" / output_group / run_tag
                if not tag_dir.exists():
                    tag_dir.mkdir(parents=True, exist_ok=False)
                    return tag_dir / "chain.hdf5"
                run_index += 1

        tag_dir = root / run_tag
        _mkdir_existing_ok(tag_dir)

        run_index = 1
        while True:
            run_dir = tag_dir / f"run_{run_index:03d}"
            if not run_dir.exists():
                run_dir.mkdir(parents=True, exist_ok=False)
                return run_dir / "chain.hdf5"
            run_index += 1

    out_file = _resolve_output_path(
        output_base,
        mcmc_cfg.get("output_file", "results/chain_universality.hdf5"),
    )
    _mkdir_existing_ok(out_file.parent)
    return out_file


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _resolve_emulator_cache_file(
    em_cfg: dict,
    task_dir: Path,
    system: str,
    n_pc: int,
) -> Path | None:
    """Resolve the analysis-local emulator cache path.

    Emulators are analysis products, not reusable experiment data. They are
    read from and saved under ``<analysis_dir>/emulator`` by default, where
    ``analysis_dir`` is the parent of the task directory.  A relative
    ``emulator.cache_dir`` is also resolved from ``analysis_dir``.
    """
    if em_cfg.get("cache", True) is False:
        return None

    output_pkl = em_cfg.get("output_pkl")
    if output_pkl:
        return _resolve_output_path(task_dir, output_pkl)

    cache_dir = em_cfg.get("cache_dir", "emulator")
    n_restarts = em_cfg.get("n_restarts", 4)
    filename = em_cfg.get(
        "cache_file",
        f"emulators_{_slug(system)}_npc{n_pc}_restarts{n_restarts}.pkl",
    )
    return _resolve_output_path(task_dir.parent, cache_dir) / filename


def _plot_config_path(config_path: Path) -> Path:
    study_dir = config_path.parent.parent
    candidates = [
        study_dir / "plot" / "corner_plot_config.json",
        config_path.parent / "plot" / "corner_plot_config.json",
        config_path.parent.parent / "plot" / "corner_plot_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find corner_plot_config.json. Tried: {tried}")


def _task_key_candidates(mcmc_cfg: dict, config_path: Path, system: str) -> list[str]:
    candidates: list[str] = []
    run_tag = str(mcmc_cfg.get("run_tag", "")).strip()
    if run_tag:
        candidates.append(run_tag)

    config_stem = config_path.stem
    if config_stem and config_stem not in candidates:
        candidates.append(config_stem)

    system = str(system).strip()
    if system and system not in candidates:
        candidates.append(system)

    return candidates


def _initialize_ptemcee_walkers(
    prior: UniformPrior,
    log_likelihood_fn,
    n_walkers: int,
    n_temps: int,
    seed: int | None,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Draw starting positions that are inside the posterior support.

    ptemcee raises immediately if any walker starts at a point with ``logp = -inf``.
    For tight posteriors such as the zero-exp study, naive prior sampling can
    produce valid prior samples that still sit outside the likelihood support.
    This helper rejection-samples walkers until all are finite.
    """
    rng = np.random.default_rng(seed)
    walkers = prior.sample(n_walkers, seed=seed)

    def _finite_posterior(theta: np.ndarray) -> bool:
        return np.isfinite(prior.log_prob(theta)) and np.isfinite(log_likelihood_fn(theta))

    attempts = 0
    invalid = [idx for idx, theta in enumerate(walkers) if not _finite_posterior(theta)]
    while invalid:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Could not initialize ptemcee walkers inside posterior support after "
                f"{max_attempts} attempts. Try a different seed or relax the initial ranges."
            )

        for idx in invalid:
            walkers[idx] = rng.uniform(prior.mins, prior.maxs)
        invalid = [idx for idx, theta in enumerate(walkers) if not _finite_posterior(theta)]

    return np.tile(walkers, (n_temps, 1, 1))


def _resolve_plot_root(out_file: Path) -> Path:
    env_plots_dir = os.environ.get("HIC_BAYES_PLOTS_DIR")
    if env_plots_dir:
        return Path(os.path.expanduser(os.path.expandvars(env_plots_dir)))

    run_id_env = os.environ.get("HIC_BAYES_RUN_ID")
    run_root_env = os.environ.get("HIC_BAYES_RUN_ROOT")
    if run_id_env and run_root_env:
        return (
            Path(os.path.expanduser(os.path.expandvars(run_root_env)))
            / "plots"
            / str(run_id_env)
        )

    for idx, part in enumerate(out_file.parts):
        if re.fullmatch(r"run_\d{3}", part) and idx >= 2 and out_file.parts[idx - 1] == "results":
            study_root = Path(*out_file.parts[: idx - 1])
            return study_root / "plots" / part

    return out_file.parent / "plots"


def _generate_individual_plots(
    *,
    config_path: Path,
    mcmc_cfg: dict,
    system: str,
    out_file: Path,
) -> None:
    plot_config_path = _plot_config_path(config_path)
    with open(plot_config_path) as f:
        plot_cfg = json.load(f)

    experiments = plot_cfg.get("experiments", {})
    jobs = plot_cfg.get("individual_plots", {}).get("jobs", [])
    task_key = next((key for key in _task_key_candidates(mcmc_cfg, config_path, system) if key in experiments), None)
    if task_key is None:
        print("  Skipping automatic individual plots: no matching experiment key found.")
        return

    exp_cfg = dict(experiments[task_key])
    exp_cfg["path"] = str(out_file.parent)

    filtered_jobs = [job for job in jobs if job.get("experiment") == task_key]
    if not filtered_jobs:
        print("  Skipping automatic individual plots: no matching individual plot job found.")
        return

    plot_root = _resolve_plot_root(out_file)
    filtered_cfg = {
        "global": {
            **plot_cfg.get("global", {}),
            "analysis_root": ".",
            "results_root": ".",
            "output_root": str(plot_root),
        },
        "parameter_labels": plot_cfg.get("parameter_labels", []),
        "parameter_groups": plot_cfg.get("parameter_groups", {}),
        "experiments": {task_key: exp_cfg},
        "individual_plots": {
            "output_subdir": plot_cfg.get("individual_plots", {}).get("output_subdir", "individual"),
            "jobs": filtered_jobs,
        },
        "comparison_plots": {
            "output_subdir": plot_cfg.get("comparison_plots", {}).get("output_subdir", "comparisons"),
            "jobs": [],
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(filtered_cfg, f, indent=2)
        f.write("\n")
        tmp_config = Path(f.name)

    try:
        print(f"  Generating automatic individual plots from: {plot_config_path}")
        cmd = [sys.executable, "-m", "hic_bayes.plot", "--config", str(tmp_config), "--mode", "individual"]
        subprocess.run(cmd, check=True)
    finally:
        try:
            tmp_config.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# PCA  (mirrors the SVD convention of Bayesian_pipeline.py)
# ---------------------------------------------------------------------------

def fit_svd_pca(Y_flat: np.ndarray, n_pc: int) -> tuple:
    """Fit StandardScaler + truncated SVD (same convention as Bayesian_pipeline).

    Returns
    -------
    scaler : StandardScaler  — fitted to Y_flat
    pc_scores : array (n_design, n_pc)   — whitened PC scores
    inverse_tf_matrix : array (n_pc, n_obs)  — maps PC scores → observable space
        y_hat = z @ A + scaler.mean_
    """
    scaler = StandardScaler(copy=True)
    Y_scaled = scaler.fit_transform(Y_flat)         # (N, n_obs)

    u, s, vh = np.linalg.svd(Y_scaled, full_matrices=True)
    N = u.shape[0]

    # Whitened PC scores  (N, n_pc) — same as u[:,0:n_pc] * sqrt(N-1)
    pc_scores = u[:, :n_pc] * math.sqrt(N - 1)

    # Inverse-transform matrix  (n_pc, n_obs)
    # y_hat = pc_scores @ A + scaler.mean_
    A = np.diag(s[:n_pc]) @ vh[:n_pc, :] * scaler.scale_.reshape(1, -1) / math.sqrt(N - 1)

    return scaler, pc_scores, A


def _impute_simulation_blocks(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Impute remaining NaN/inf values within each observable block.

    The legacy qgp-parameter-inference code did this for non-removed design
    points: if a single observable entry was NaN, it was replaced by the mean of
    the remaining values from that same observable at the same design point.
    """
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


def emulator_predict_obs(
    theta: np.ndarray,
    emulators: list,
    inverse_tf_matrix: np.ndarray,
    scaler_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict observable-space mean and covariance from the GP emulators.

    Parameters
    ----------
    theta : array, shape (n_params,)
    emulators : list of fitted sklearn GaussianProcessRegressor (one per PC)
    inverse_tf_matrix : array, shape (n_pc, n_obs)
    scaler_mean : array, shape (n_obs,)
        Mean vector from the fitted StandardScaler.

    Returns
    -------
    mean : array, shape (n_obs,)
    cov_emulator : array, shape (n_obs, n_obs)
        Propagated emulator uncertainty in observable space.
    """
    theta_2d = np.array(theta).reshape(1, -1)
    n_pc = len(emulators)
    pc_means = np.zeros(n_pc)
    pc_vars  = np.zeros(n_pc)
    for i, gp in enumerate(emulators):
        mn, std = gp.predict(theta_2d, return_std=True)
        pc_means[i] = mn.flat[0]
        pc_vars[i]  = std.flat[0] ** 2

    A = inverse_tf_matrix[:n_pc, :]                # (n_pc, n_obs)
    mean = pc_means @ A + scaler_mean               # (n_obs,)
    cov_emulator = A.T @ np.diag(pc_vars) @ A      # (n_obs, n_obs)
    return mean, cov_emulator


# ---------------------------------------------------------------------------
# Emulator training (sklearn GP per PC — identical kernel to original pipeline)
# ---------------------------------------------------------------------------

def _train_one_gp(args):
    """Top-level function for multiprocessing Pool (must be picklable)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    design_vals, z_col, ptp, n_restarts = args
    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=ptp, length_scale_bounds=np.outer(ptp, (0.4, 1e2)))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1e2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        alpha=1e-10,
    )
    gp.fit(design_vals, z_col.reshape(-1, 1))
    return gp


def train_emulators(
    design: np.ndarray,
    pc_scores: np.ndarray,
    prior: UniformPrior,
    n_jobs: int = 1,
    n_restarts: int = 4,
) -> list:
    """Train one sklearn GP per PC column.

    Parameters
    ----------
    design : array, shape (n_design, n_params)
    pc_scores : array, shape (n_design, n_pc)
    prior : UniformPrior  — used to extract parameter ranges
    n_jobs : int
    """
    ptp = prior.ranges[:, 1] - prior.ranges[:, 0]
    n_pc = pc_scores.shape[1]
    args = [(design, pc_scores[:, i], ptp, n_restarts) for i in range(n_pc)]
    if n_jobs != 1:
        with Pool(n_jobs) as pool:
            emulators = pool.map(_train_one_gp, args)
    else:
        emulators = [_train_one_gp(a) for a in args]
    return emulators


# ---------------------------------------------------------------------------
# Pickle-safe log-likelihood callable (required by multiprocessing workers)
# ---------------------------------------------------------------------------

class ObsSpaceLogLikelihood:
    """Observable-space log-likelihood for ptemcee.

    This class is defined at module scope so it can be pickled and sent to
    multiprocessing workers when ``threads > 1``.
    """

    def __init__(
        self,
        emulators: list,
        inverse_tf_matrix: np.ndarray,
        scaler_mean: np.ndarray,
        exp_mean: np.ndarray,
        exp_cov: np.ndarray,
        regularize_covariance: bool = False,
        covariance_jitter_eps: float = 1e-8,
    ) -> None:
        self.emulators = emulators
        self.inverse_tf_matrix = inverse_tf_matrix
        self.scaler_mean = scaler_mean
        self.exp_mean = exp_mean
        self.exp_cov = exp_cov
        self.regularize_covariance = regularize_covariance
        self.covariance_jitter_eps = covariance_jitter_eps
        self.log_covariance = str(os.environ.get("HIC_BAYES_LOG_COVARIANCE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._covariance_logged = False

    def _log_covariance_diagnostics(self, cov_em: np.ndarray, total_cov: np.ndarray) -> None:
        if not self.log_covariance or self._covariance_logged:
            return

        exp_cov = self.exp_cov
        print("\nCovariance diagnostics (first likelihood call):")
        print(
            f"  emulator     : trace={np.trace(cov_em):.3e}  "
            f"diag_mean={np.mean(np.diag(cov_em)):.3e}  fro={np.linalg.norm(cov_em):.3e}"
        )
        print(
            f"  experimental : trace={np.trace(exp_cov):.3e}  "
            f"diag_mean={np.mean(np.diag(exp_cov)):.3e}  fro={np.linalg.norm(exp_cov):.3e}"
        )
        print(
            f"  total        : trace={np.trace(total_cov):.3e}  "
            f"diag_mean={np.mean(np.diag(total_cov)):.3e}  fro={np.linalg.norm(total_cov):.3e}"
        )
        print(
            f"  ratio exp/emu (fro): {np.linalg.norm(exp_cov) / max(np.linalg.norm(cov_em), 1e-300):.3e}"
        )
        self._covariance_logged = True

    def __call__(self, theta: np.ndarray) -> float:
        mn, cov_em = emulator_predict_obs(
            theta,
            self.emulators,
            self.inverse_tf_matrix,
            self.scaler_mean,
        )
        total_cov = build_total_covariance_obs(cov_em, self.exp_cov)
        # Numerical guard: enforce symmetry before factorization.
        total_cov = 0.5 * (total_cov + total_cov.T)
        self._log_covariance_diagnostics(cov_em, total_cov)
        residuals = self.exp_mean - mn

        if not self.regularize_covariance:
            return mvn_loglike(residuals, total_cov)

        # In none-mode, exp covariance is zero and emulator covariance can be
        # rank-deficient (n_pc << n_obs). Add adaptive diagonal jitter until
        # Cholesky succeeds or we exhaust a conservative jitter ladder.
        diag_mean = float(np.mean(np.diag(total_cov)))
        if not np.isfinite(diag_mean):
            diag_mean = 1.0
        scale = max(diag_mean, 1.0)

        eye = np.eye(total_cov.shape[0])
        base_jitter = self.covariance_jitter_eps * scale
        jitter = base_jitter
        for _ in range(7):
            ll = mvn_loglike(residuals, total_cov + jitter * eye)
            if np.isfinite(ll):
                return ll
            jitter *= 10.0

        return -np.inf


# ---------------------------------------------------------------------------
# Time estimation
# ---------------------------------------------------------------------------

def estimate_throughput(n_threads: int, n_params: int) -> float:
    """Estimate likelihood evals/sec based on hardware and problem size.

    Uses benchmarks from typical runs. For a baseline:
      - 1 thread, 17 params  → ~200 evals/sec (single-threaded overhead)
      - 20 threads, 17 params → ~1750 evals/sec (from real measurement)

    Heuristic: throughput ≈ baseline × sqrt(threads) × (1 / (1 + 0.01 × extra_params))
    """
    baseline_1t_17p = 200.0  # measured single-thread baseline
    baseline_20t_17p = 1750.0
    measured_speedup_20t = baseline_20t_17p / baseline_1t_17p  # ~8.75x

    # Simple scaling: sqrt for sub-linear parallel efficiency
    actual_speedup = 1.0 if n_threads == 1 else np.sqrt(n_threads) * 0.85
    base_est = baseline_1t_17p * actual_speedup

    # Penalize higher dimensionality slightly (more likelihood eval overhead)
    param_penalty = 1.0 / (1.0 + 0.005 * max(0, n_params - 17))

    return base_est * param_penalty


def format_duration(seconds: float) -> str:
    """Format seconds into [H]h [M]m [S]s notation."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def _parameter_labels(params: list[dict]) -> list[str]:
    """Return Matplotlib-ready parameter labels from YAML config entries."""
    return [
        str(p.get("label", p["name"])).replace("\\\\", "\\")
        for p in params
    ]


def _summarize_and_plot_chain(
    chain: np.ndarray,
    log_prob: np.ndarray,
    burnin: int,
    thin: int,
    param_names: list[str],
    param_labels: list[str],
    system: str,
    out_file: Path,
) -> None:
    # -------------------------------------------------------- posterior summary
    flat_chain    = thin_chain(chain, burnin, thin)
    flat_log_prob = log_prob[:, burnin::thin].ravel()
    print(f"\nPosterior: {flat_chain.shape[0]} samples after burn-in / thinning")

    summary = posterior_summary(flat_chain, param_names)
    print(f"\n{'Parameter':<22}  {'Mean':>10}  {'Std':>8}  {'CI5':>10}  {'CI95':>10}")
    print("  " + "-" * 64)
    for name, s in summary.items():
        print(
            f"  {name:<20}  {s['mean']:>10.4f}  {s['std']:>8.4f}"
            f"  {s['ci_low']:>10.4f}  {s['ci_high']:>10.4f}"
        )

    theta_map = map_estimate(flat_chain, flat_log_prob)
    print(f"\nMAP estimate:\n  {dict(zip(param_names, theta_map))}")

    # -------------------------------------------------- save posterior CSV
    _mkdir_existing_ok(out_file.parent)

    csv_path = out_file.with_suffix(".csv")
    pd.DataFrame(flat_chain, columns=param_names).to_csv(csv_path, index=False)
    print(f"\nPosterior chain saved to: {csv_path}")

    # -------------------------------------------------- walker trace plot
    trace_fig = plot_trace(chain, param_labels, burnin=burnin)
    trace_path = out_file.parent / "walkers_trace.pdf"
    trace_fig.savefig(str(trace_path), bbox_inches="tight")
    print(f"Trace plot saved to     : {trace_path}")

    # -------------------------------------------------- corner plot
    corner_fig = corner_plot(flat_chain, param_labels, title=system)
    corner_path = out_file.parent / "posterior_corner.pdf"
    corner_fig.savefig(str(corner_path), bbox_inches="tight")
    print(f"Corner plot saved to    : {corner_path}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="hic-bayes Universality analysis — observable-space likelihood"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--retrain-emulators",
        action="store_true",
        default=False,
        help="Force retraining of GP emulators even if a pkl file is present.",
    )
    parser.add_argument(
        "--plot-only-from",
        help=(
            "Load an existing chain HDF5 and regenerate posterior CSV/plots "
            "without retraining emulators or running MCMC."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = _load_config(args.config)
    task_dir = config_path.parent
    output_base = _output_base_dir(config_path)
    system      = cfg.get("system", "UniversalityPbPb2760")
    description = cfg.get("description", system)
    print(f"\n=== hic-bayes Universality | {description} ===\n")

    params       = cfg["parameters"]
    param_names  = [p["name"] for p in params]
    param_labels = _parameter_labels(params)
    mcmc_cfg     = cfg["mcmc"]
    burnin       = mcmc_cfg.get("burnin", 500)
    thin         = mcmc_cfg.get("thin", 10)

    plot_only_from = args.plot_only_from or mcmc_cfg.get("plot_only_from")
    if plot_only_from:
        chain_file = _resolve_output_path(output_base, plot_only_from)
        print(f"Loading existing chain for plot-only mode:\n  {chain_file}")
        chain, log_prob = load_chain_hdf5(str(chain_file))
        _summarize_and_plot_chain(
            chain=chain,
            log_prob=log_prob,
            burnin=burnin,
            thin=thin,
            param_names=param_names,
            param_labels=param_labels,
            system=system,
            out_file=chain_file,
        )
        return

    # ------------------------------------------------------------------ data
    dc       = cfg["data"]
    conv_cfg = cfg.get("conversion", {})

    design_df   = read_table(cfg, "design", kind="model")
    sim_df      = read_table(cfg, "simulation", kind="model")
    exp_df      = read_table(cfg, "experiment", kind="data")
    err_df      = read_table(cfg, "exp_error", kind="data")

    observable_columns = conv_cfg.get("observable_columns") or []
    if observable_columns:
        missing_columns = {
            "simulation": [col for col in observable_columns if col not in sim_df.columns],
            "experiment": [col for col in observable_columns if col not in exp_df.columns],
            "exp_error": [col for col in observable_columns if col not in err_df.columns],
        }
        missing_columns = {name: cols for name, cols in missing_columns.items() if cols}
        if missing_columns:
            missing_msg = "; ".join(
                f"{name} missing {len(cols)} columns" for name, cols in missing_columns.items()
            )
            raise ValueError(
                "conversion.observable_columns could not be aligned across runtime tables: "
                f"{missing_msg}."
            )

        sim_df = sim_df.loc[:, observable_columns].copy()
        exp_df = exp_df.loc[:, observable_columns].copy()
        err_df = err_df.loc[:, observable_columns].copy()
        print(
            f"Aligned converted runtime tables to {len(observable_columns)} configured observables."
        )

    y_exp       = exp_df.values.ravel()   # (n_obs,)
    y_err       = err_df.values.ravel()   # (n_obs,)

    design_arr  = design_df.values          # (n_design, n_params)
    sim_arr     = sim_df.values             # (n_design, n_obs)

    removed_design_idx = [int(x) for x in conv_cfg.get("design_remove_idx", [])]
    raw_design_rows = conv_cfg.get("raw_design_rows")
    if removed_design_idx and raw_design_rows is not None:
        expected_filtered = int(raw_design_rows) - len(removed_design_idx)
        if design_arr.shape[0] == int(raw_design_rows):
            keep_mask = np.ones(design_arr.shape[0], dtype=bool)
            keep_mask[removed_design_idx] = False
            design_arr = design_arr[keep_mask]
            sim_df = sim_df.loc[keep_mask].copy()
            sim_arr = sim_arr[keep_mask]
            print(
                f"Filtered {len(removed_design_idx)} legacy design points from runtime data "
                f"({raw_design_rows} -> {design_arr.shape[0]} rows)."
            )
        elif design_arr.shape[0] == expected_filtered:
            print(
                f"Runtime data already excludes {len(removed_design_idx)} legacy design points "
                f"({design_arr.shape[0]} rows)."
            )
        else:
            print(
                f"Warning: design row count {design_arr.shape[0]} does not match either the raw "
                f"({raw_design_rows}) or filtered ({expected_filtered}) legacy expectation."
            )

    if not np.isfinite(sim_arr).all():
        sim_df = _impute_simulation_blocks(sim_df)
        sim_arr = sim_df.values
        print("Imputed remaining non-finite simulation entries within observable blocks.")

    prior       = UniformPrior.from_config(params)

    print(f"Design   : {design_arr.shape[0]} points × {design_arr.shape[1]} params")
    print(f"Simulation: {sim_arr.shape[0]} × {sim_arr.shape[1]} observables")
    print(f"Prior    : {prior.n_params} parameters")

    # ------------------------------------------------------------------- PCA
    n_pc = cfg.get("pca", {}).get("n_components", 6)
    print(f"\nFitting SVD PCA with {n_pc} components …")
    scaler, pc_scores, inv_tf = fit_svd_pca(sim_arr, n_pc)
    reconstructed = pc_scores @ inv_tf + scaler.mean_
    rec_err = np.abs(reconstructed - sim_arr).max()
    print(f"  PCA reconstruction max error: {rec_err:.2e}")

    # --------------------------------------------------------------- emulator
    em_cfg      = cfg.get("emulator", {})
    cache_path = _resolve_emulator_cache_file(em_cfg, task_dir, system, n_pc)

    emulators = None
    if (
        not args.retrain_emulators
        and cache_path is not None
        and cache_path.exists()
    ):
        print(f"\nLoading task-local emulators from:\n  {cache_path}")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", InconsistentVersionWarning)
                with open(cache_path, "rb") as f:
                    emulators = pickle.load(f)

            has_version_warning = any(
                issubclass(w.category, InconsistentVersionWarning)
                for w in caught
            )
            allow_mismatch = bool(em_cfg.get("allow_version_mismatch", False))

            if has_version_warning and not allow_mismatch:
                print(
                    "  Detected sklearn version mismatch in emulator pickle.\n"
                    "  Retraining emulators with current sklearn for consistency.\n"
                    "  Set emulator.allow_version_mismatch: true in YAML to force"
                    " loading this pickle as-is."
                )
                emulators = None
            elif has_version_warning and allow_mismatch:
                print(
                    "  Warning: sklearn version mismatch detected, but continuing "
                    "because emulator.allow_version_mismatch=true."
                )

            if emulators is not None:
                print(f"  Loaded {len(emulators)} GP emulators.")
        except Exception as exc:
            print(f"  Failed to load emulator pickle ({exc}).")
            emulators = None

    if emulators is None:
        n_jobs = em_cfg.get("n_jobs", 1)
        n_restarts = int(em_cfg.get("n_restarts", 4))
        print(
            f"\nTraining {n_pc} GP emulators "
            f"(n_jobs={n_jobs}, n_restarts={n_restarts}) …"
        )
        t0 = time.time()
        emulators = train_emulators(
            design_arr,
            pc_scores,
            prior,
            n_jobs=n_jobs,
            n_restarts=n_restarts,
        )
        print(f"  Done in {time.time()-t0:.1f} s")
        if cache_path is not None:
            _mkdir_existing_ok(cache_path.parent)
            tmp_cache_path = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.tmp"
            )
            with open(tmp_cache_path, "wb") as f:
                pickle.dump(emulators, f)
            os.replace(tmp_cache_path, cache_path)
            print(f"  Saved emulator cache to {cache_path}")

    # ------------------------------------------------------ experimental cov
    exp_cov_cfg  = cfg.get("experimental_covariance", {})
    exp_cov_mode = str(exp_cov_cfg.get("mode", "diag")).strip().lower()
    exp_cov_rho  = float(exp_cov_cfg.get("corr_rho", 0.0))
    exp_cov_length = exp_cov_cfg.get("corr_length", None)
    exp_cov_length = None if exp_cov_length is None else float(exp_cov_length)
    covariance_jitter_eps = float(exp_cov_cfg.get("none_mode_jitter_eps", 1e-8))
    no_exp_cov_modes = {"none", "zero", "nocov", "noexp", "absent"}
    print(f"\nExperimental covariance: mode={exp_cov_mode!r}  rho={exp_cov_rho}")
    if exp_cov_length is not None:
        print(f"  Correlation length: {exp_cov_length}")
    if exp_cov_mode in no_exp_cov_modes:
        print(f"  Covariance regularization (none mode): eps={covariance_jitter_eps:g}")

    exp_variance = y_err ** 2
    exp_cov = build_experimental_covariance(
        exp_variance,
        mode=exp_cov_mode,
        corr_rho=exp_cov_rho,
        corr_length=exp_cov_length,
    )

    # ------------------------------------------------------------------ MCMC
    sampler_name = str(mcmc_cfg.get("sampler", "ptemcee")).strip().lower()
    n_walkers  = mcmc_cfg["n_walkers"]
    n_steps    = mcmc_cfg["n_steps"]
    n_temps    = mcmc_cfg.get("n_temps", 20)
    # HPC override: HIC_BAYES_NTHREADS env var takes precedence over the YAML
    # so the YAML can be committed with n_threads: 1 (macOS-safe) while the
    # SLURM job sets the env var to the number of allocated CPUs.
    import os as _os
    n_threads  = int(_os.environ.get("HIC_BAYES_NTHREADS", mcmc_cfg.get("n_threads", 1)))
    out_file   = _resolve_output_file(mcmc_cfg, output_base)
    seed       = mcmc_cfg.get("seed", None)

    # ---- Validate MCMC configuration ----
    min_walkers = 2 * prior.n_params + 1
    if sampler_name == "ptemcee" and n_walkers <= 2 * prior.n_params:
        raise ValueError(
            f"ptemcee constraint violated: n_walkers must be > 2 × n_params\n"
            f"  Current: n_walkers={n_walkers}, n_params={prior.n_params}\n"
            f"  Minimum: n_walkers > {2 * prior.n_params}\n"
            f"  Recommended: n_walkers >= {min_walkers}\n"
            f"  Fix: Update mcmc.n_walkers in the YAML config to at least {min_walkers}."
        )
    elif sampler_name == "emcee" and n_walkers < 2 * prior.n_params:
        raise ValueError(
            f"emcee constraint violated: n_walkers must be >= 2 × n_params\n"
            f"  Current: n_walkers={n_walkers}, n_params={prior.n_params}\n"
            f"  Minimum: n_walkers >= {2 * prior.n_params}\n"
            f"  Fix: Update mcmc.n_walkers in the YAML config to at least {2 * prior.n_params}."
        )

    log_ll = ObsSpaceLogLikelihood(
        emulators=emulators,
        inverse_tf_matrix=inv_tf,
        scaler_mean=scaler.mean_,
        exp_mean=y_exp,
        exp_cov=exp_cov,
        regularize_covariance=(exp_cov_mode in no_exp_cov_modes),
        covariance_jitter_eps=covariance_jitter_eps,
    )

    rng = np.random.default_rng(seed)
    p0 = prior.sample(n_walkers, seed=seed)

    # ---- MCMC setup summary ----
    n_ll_evals = n_temps * n_walkers * n_steps
    post_burn_steps = (n_steps - burnin) // thin
    final_samples = n_walkers * post_burn_steps
    print(
        f"\n{'=' * 70}"
        f"\n{sampler_name} MCMC\n"
        f"{'=' * 70}\n"
        f"  Walkers:           {n_walkers:6d}\n"
        f"  Steps:             {n_steps:6d}  (burn-in={burnin}, thin={thin})\n"
        f"  Temperatures:      {n_temps:6d}\n"
        f"  Threads:           {n_threads:6d}\n"
        f"  Likelihood evals:  {n_ll_evals:6d}  ({n_temps} temps × {n_walkers} walkers × {n_steps} steps)\n"
        f"  Final samples:     {final_samples:6d}\n"
        f"{'-' * 70}\n"
    )

    # Estimate runtime
    est_throughput = estimate_throughput(n_threads, prior.n_params)
    est_seconds = n_ll_evals / est_throughput
    est_duration = format_duration(est_seconds)
    print(
        f"  Estimated throughput: ~{est_throughput:>7.0f} evals/sec\n"
        f"  Estimated duration:   ~{est_duration:>8s}\n"
        f"{'-' * 70}\n"
        f"  Progress bar below will update per completed step.\n"
        f"{'=' * 70}\n"
    )

    t0 = time.time()
    if sampler_name == "emcee":
        def log_prob(theta: np.ndarray) -> float:
            lp = prior.log_prob(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_ll(theta)

        chain, log_prob = run_emcee(
            log_prob,
            n_walkers,
            n_steps,
            prior.n_params,
            p0,
            output_file=out_file,
            n_jobs=n_threads,
            progress=True,
        )
    elif sampler_name == "ptemcee":
        p0_pt = _initialize_ptemcee_walkers(
            prior,
            log_ll,
            n_walkers=n_walkers,
            n_temps=n_temps,
            seed=seed,
        )
        chain, log_prob = run_ptemcee(
            log_ll,
            prior.log_prob,
            n_walkers,
            n_steps,
            prior.n_params,
            p0_pt,
            n_temps=n_temps,
            output_file=out_file,
            n_jobs=n_threads,
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler_name!r}. Use 'emcee' or 'ptemcee'.")
    elapsed = time.time() - t0
    ll_per_sec = n_ll_evals / elapsed
    actual_duration = format_duration(elapsed)

    # Compare with estimate
    error_pct = 100.0 * (elapsed - est_seconds) / est_seconds
    error_sign = "+" if error_pct > 0 else ""
    print(
        f"\n{'=' * 70}\n"
        f"Sampling completed in {actual_duration}\n"
        f"  Wall time:    {elapsed:.1f} s\n"
        f"  Throughput:   {ll_per_sec:>8.0f} likelihood evals/sec\n"
        f"  Estimate was: {est_duration} ({error_sign}{error_pct:.1f}%)\n"
        f"{'=' * 70}"
    )

    _summarize_and_plot_chain(
        chain=chain,
        log_prob=log_prob,
        burnin=burnin,
        thin=thin,
        param_names=param_names,
        param_labels=param_labels,
        system=system,
        out_file=Path(out_file),
    )

    _generate_individual_plots(
        config_path=config_path,
        mcmc_cfg=mcmc_cfg,
        system=system,
        out_file=Path(out_file),
    )


if __name__ == "__main__":
    main()
