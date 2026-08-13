"""Command-line entry point for multi-fidelity (transfer-learning) Bayesian analysis.

Requires the optional multifidelity dependencies::

    pip install hic-bayes[multifidelity]

Usage
-----
    python -m pipelines.run_tl_analysis config/example_tl_AuAu200.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from hic_bayes.calibration import calibration_report
from hic_bayes.emulator_mf import MultiFidelityEmulator
from hic_bayes.io import read_array
from hic_bayes.likelihood import build_covariance, mvn_loglike
from hic_bayes.mcmc import run_emcee, thin_chain
from hic_bayes.pca import fit_pca
from hic_bayes.posterior import corner_plot, map_estimate, posterior_summary
from hic_bayes.prior import UniformPrior


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hic-bayes multi-fidelity: transfer-learning GP emulator"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    print(f"\n=== hic-bayes TL | {cfg.get('description', cfg.get('system', ''))} ===\n")

    params = cfg["parameters"]
    param_names = [p["name"] for p in params]
    param_labels = [p.get("label", p["name"]) for p in params]
    prior = UniformPrior.from_config(params)

    # ------------------------------------------------------------------ data
    dc = cfg["data"]
    design_low = read_array(cfg, "design", kind="model")       # same design for both fidelities
    design_high = read_array(cfg, "design", kind="model")

    sim_low = read_array(cfg, "simulation_low", kind="model")
    sim_high = read_array(cfg, "simulation_high", kind="model")
    y_exp = read_array(cfg, "experiment", kind="data").ravel()
    y_err = read_array(cfg, "exp_error", kind="data").ravel()

    # Optional sub-sampling of high-fidelity points (key TL use case)
    n_high = cfg.get("n_high_fidelity_points")
    if n_high is not None:
        rng = np.random.default_rng(cfg.get("seed", 42))
        idx = rng.choice(len(design_high), size=n_high, replace=False)
        design_high = design_high[idx]
        sim_high = sim_high[idx]
        print(f"Using {n_high} high-fidelity points (subsampled from {len(read_array(cfg, 'design', kind='model'))}).")

    # ------------------------------------------------------------------- PCA
    # PCA fitted on low-fidelity data; high-fidelity data projected onto same basis
    n_pcs = cfg["pca"]["n_components"]
    pca = fit_pca(sim_low, n_components=n_pcs)
    Z_low = pca.transform(sim_low)
    Z_high = pca.transform(sim_high)
    print(
        f"PCA (low-fidelity basis): {n_pcs} PCs explain "
        f"{pca.cumulative_variance[-1]:.1%} of variance"
    )

    exp_variance_pc = np.abs(pca.transform(y_err.reshape(1, -1))).ravel() ** 2

    # --------------------------------------------------------------- emulator
    print("Training multi-fidelity emulator …")
    emulator = MultiFidelityEmulator(
        n_optimization_restarts=cfg.get("emulator", {}).get("n_restarts", 5)
    )
    emulator.fit(design_low, Z_low, design_high, Z_high)
    print("Emulator training complete.\n")

    # ------------------------------------------------------------------ MCMC
    mcmc_cfg = cfg["mcmc"]
    n_params = prior.n_params
    n_walkers = mcmc_cfg["n_walkers"]
    n_steps = mcmc_cfg["n_steps"]
    out_file = mcmc_cfg.get("output_file", "results/chain_tl.hdf5")

    y_exp_pc = pca.transform(y_exp.reshape(1, -1)).ravel()

    def log_prob(theta: np.ndarray) -> float:
        lp = prior.log_prob(theta)
        if not np.isfinite(lp):
            return -np.inf
        y_pred_pc, y_std_pc = emulator.predict(theta.reshape(1, -1), return_std=True)
        cov = build_covariance(y_std_pc.ravel() ** 2, exp_variance_pc)
        return lp + mvn_loglike(y_exp_pc - y_pred_pc.ravel(), cov)

    p0 = prior.sample(n_walkers)
    print(f"Running emcee ({n_walkers} walkers × {n_steps} steps) …")
    chain, log_prob_arr = run_emcee(
        log_prob, n_walkers, n_steps, n_params, p0, output_file=out_file
    )

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

    theta_map = map_estimate(flat_chain, flat_log_prob)
    print(f"\nMAP estimate: {dict(zip(param_names, theta_map))}")

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig = corner_plot(flat_chain, param_labels, title=f"{cfg.get('system', '')} [TL]")
    corner_path = str(Path(out_file).parent / "posterior_corner_tl.pdf")
    fig.savefig(corner_path, bbox_inches="tight")
    print(f"\nCorner plot → {corner_path}\nDone.")


if __name__ == "__main__":
    main()
