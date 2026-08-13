"""Command-line entry point for a standard single-fidelity Bayesian analysis.

Usage
-----
    # From the hic-bayes root directory:
    python -m pipelines.run_analysis config/example_PbPb2760.yaml

    # Or after pip install -e .:
    hic-bayes config/example_PbPb2760.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from hic_bayes.calibration import calibration_report
from hic_bayes.emulator import Emulator
from hic_bayes.io import read_array
from hic_bayes.likelihood import build_covariance, mvn_loglike
from hic_bayes.mcmc import run_emcee, run_ptemcee, thin_chain
from hic_bayes.pca import fit_pca
from hic_bayes.plots import plot_trace, plot_variance_explained
from hic_bayes.posterior import corner_plot, map_estimate, posterior_summary
from hic_bayes.prior import UniformPrior


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_log_prob(
    emulator: Emulator,
    pca,
    prior: UniformPrior,
    y_exp: np.ndarray,
    exp_variance: np.ndarray,
) -> callable:
    """Return a log-probability function for use with MCMC samplers."""
    y_exp_pc = pca.transform(y_exp.reshape(1, -1)).ravel()

    def log_prob(theta: np.ndarray) -> float:
        lp = prior.log_prob(theta)
        if not np.isfinite(lp):
            return -np.inf
        y_pred_pc, y_std_pc = emulator.predict(
            theta.reshape(1, -1), return_std=True
        )
        cov = build_covariance(
            y_std_pc.ravel() ** 2,
            exp_variance,
        )
        return lp + mvn_loglike(y_exp_pc - y_pred_pc.ravel(), cov)

    return log_prob


def build_log_likelihood(
    emulator: Emulator,
    pca,
    y_exp: np.ndarray,
    exp_variance: np.ndarray,
) -> callable:
    """Return a separable log-likelihood for ptemcee (no prior term)."""
    y_exp_pc = pca.transform(y_exp.reshape(1, -1)).ravel()

    def log_likelihood(theta: np.ndarray) -> float:
        y_pred_pc, y_std_pc = emulator.predict(
            theta.reshape(1, -1), return_std=True
        )
        cov = build_covariance(y_std_pc.ravel() ** 2, exp_variance)
        return mvn_loglike(y_exp_pc - y_pred_pc.ravel(), cov)

    return log_likelihood


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hic-bayes: Bayesian parameter estimation for heavy-ion models"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    print(f"\n=== hic-bayes | {cfg.get('description', cfg.get('system', ''))} ===\n")

    # ------------------------------------------------------------------ data
    dc = cfg["data"]
    design = read_array(cfg, "design", kind="model")
    sim = read_array(cfg, "simulation", kind="model")
    y_exp = read_array(cfg, "experiment", kind="data").ravel()
    y_err = read_array(cfg, "exp_error", kind="data").ravel()

    params = cfg["parameters"]
    param_names = [p["name"] for p in params]
    param_labels = [p.get("label", p["name"]) for p in params]
    prior = UniformPrior.from_config(params)

    # ------------------------------------------------------------------- PCA
    n_pcs = cfg["pca"]["n_components"]
    pca = fit_pca(sim, n_components=n_pcs)
    Z_train = pca.transform(sim)
    print(
        f"PCA: {n_pcs} PCs explain "
        f"{pca.cumulative_variance[-1]:.1%} of variance"
    )

    # Project experimental error into PC space (diagonal approximation)
    exp_variance_pc = np.abs(
        pca.transform(y_err.reshape(1, -1))
    ).ravel() ** 2

    # --------------------------------------------------------------- emulator
    em_cfg = cfg.get("emulator", {})
    emulator = Emulator(n_restarts=em_cfg.get("n_restarts", 4))
    print("Training emulator …")
    emulator.fit(
        design, Z_train, prior.ranges, n_jobs=em_cfg.get("n_jobs", -1)
    )
    print("Emulator training complete.\n")

    # ------------------------------------------------------------------ MCMC
    mcmc_cfg = cfg["mcmc"]
    n_params = prior.n_params
    n_walkers = mcmc_cfg["n_walkers"]
    n_steps = mcmc_cfg["n_steps"]
    out_file = mcmc_cfg.get("output_file", "results/chain.hdf5")
    sampler_name = mcmc_cfg.get("sampler", "emcee")

    p0 = prior.sample(n_walkers)

    if sampler_name == "emcee":
        log_prob_fn = build_log_prob(
            emulator, pca, prior, y_exp, exp_variance_pc
        )
        print(f"Running emcee ({n_walkers} walkers × {n_steps} steps) …")
        chain, log_prob_arr = run_emcee(
            log_prob_fn,
            n_walkers,
            n_steps,
            n_params,
            p0,
            output_file=out_file,
        )

    elif sampler_name == "ptemcee":
        n_temps = mcmc_cfg.get("n_temps", 10)
        log_ll_fn = build_log_likelihood(emulator, pca, y_exp, exp_variance_pc)
        p0_pt = np.tile(p0, (n_temps, 1, 1))
        print(
            f"Running ptemcee ({n_walkers} walkers × {n_steps} steps "
            f"× {n_temps} temps) …"
        )
        chain, log_prob_arr = run_ptemcee(
            log_ll_fn,
            prior.log_prob,
            n_walkers,
            n_steps,
            n_params,
            p0_pt,
            n_temps=n_temps,
            output_file=out_file,
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler_name!r}. Use 'emcee' or 'ptemcee'.")

    # -------------------------------------------------------- posterior summary
    burnin = mcmc_cfg.get("burnin", n_steps // 2)
    thin = mcmc_cfg.get("thin", 1)
    flat_chain = thin_chain(chain, burnin, thin)
    flat_log_prob = log_prob_arr[:, burnin::thin].ravel()

    summary = posterior_summary(flat_chain, param_names)
    print("\nPosterior summary (90% CI):")
    print(f"  {'Parameter':<20}  {'Mean':>10}  {'Std':>8}  {'CI low':>10}  {'CI high':>10}")
    print("  " + "-" * 62)
    for name, s in summary.items():
        print(
            f"  {name:<20}  {s['mean']:>10.4f}  {s['std']:>8.4f}"
            f"  {s['ci_low']:>10.4f}  {s['ci_high']:>10.4f}"
        )

    # MAP estimate
    theta_map = map_estimate(flat_chain, flat_log_prob)
    print(f"\nMAP estimate: {dict(zip(param_names, theta_map))}")

    # Save corner plot
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    corner_fig = corner_plot(
        flat_chain, param_labels, title=cfg.get("system", "")
    )
    corner_path = str(Path(out_file).parent / "posterior_corner.pdf")
    corner_fig.savefig(corner_path, bbox_inches="tight")
    print(f"\nCorner plot → {corner_path}")

    # Closure test
    ct_cfg = cfg.get("closure_test", {})
    if ct_cfg.get("enabled") and (
        "validation_design" in cfg.get("model", {}) or "validation_design" in dc
    ):
        idx = ct_cfg.get("design_index", 0)
        true_params = read_array(cfg, "validation_design", kind="model")[idx]
        report = calibration_report(true_params, flat_chain, param_names)
        print(
            f"\nClosure test (true params = design point {idx}):\n"
            f"  Mahalanobis distance : {report['mahalanobis']:.3f}"
            f"  (expect ~ {n_params} for well-calibrated posterior)\n"
            f"  Energy Score         : {report['energy_score']:.4f}\n"
            f"  KNN log-density      : {report['log_density']:.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
