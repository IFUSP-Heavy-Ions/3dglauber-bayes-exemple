"""Joint Bayesian analysis across two systems with shared parameters.

This pipeline reproduces the "joint posterior" idea from jetscape-emulator:
a single parameter vector is inferred by summing the log-likelihood
contributions from two systems.

Usage
-----
    python -m pipelines.run_joint_analysis \
        config/qpi_converted_Au-Au-200.yaml \
        config/qpi_converted_d-Au-200.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hic_bayes.emulator import Emulator
from hic_bayes.io import read_array
from hic_bayes.likelihood import build_covariance, mvn_loglike
from hic_bayes.mcmc import run_emcee, run_ptemcee, thin_chain
from hic_bayes.pca import fit_pca
from hic_bayes.posterior import corner_plot, map_estimate, posterior_summary
from hic_bayes.prior import UniformPrior


@dataclass
class SystemModel:
    """Container for one system's data/projection/emulator stack."""

    name: str
    config_path: Path
    prior: UniformPrior
    param_names: list[str]
    param_labels: list[str]
    y_exp_pc: np.ndarray
    exp_variance_pc: np.ndarray
    emulator: Emulator


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _system_name(cfg: dict, cfg_path: Path) -> str:
    conv_name = cfg.get("conversion", {}).get("source_system")
    if conv_name:
        return str(conv_name)
    if cfg.get("system"):
        return str(cfg["system"])
    return cfg_path.stem


def _build_system_model(cfg_path: str, n_jobs: int) -> tuple[SystemModel, dict]:
    cfg_file = Path(cfg_path)
    cfg = _load_config(str(cfg_file))

    design = read_array(cfg, "design", kind="model")
    sim = read_array(cfg, "simulation", kind="model")
    y_exp = read_array(cfg, "experiment", kind="data").ravel()
    y_err = read_array(cfg, "exp_error", kind="data").ravel()

    params = cfg["parameters"]
    param_names = [p["name"] for p in params]
    param_labels = [p.get("label", p["name"]) for p in params]
    prior = UniformPrior.from_config(params)

    n_pcs = cfg.get("pca", {}).get("n_components", 10)
    pca = fit_pca(sim, n_components=n_pcs)
    z_train = pca.transform(sim)

    y_exp_pc = pca.transform(y_exp.reshape(1, -1)).ravel()
    exp_variance_pc = np.abs(pca.transform(y_err.reshape(1, -1))).ravel() ** 2

    em_cfg = cfg.get("emulator", {})
    emulator = Emulator(n_restarts=em_cfg.get("n_restarts", 4))
    emulator.fit(design, z_train, prior.ranges, n_jobs=n_jobs)

    model = SystemModel(
        name=_system_name(cfg, cfg_file),
        config_path=cfg_file,
        prior=prior,
        param_names=param_names,
        param_labels=param_labels,
        y_exp_pc=y_exp_pc,
        exp_variance_pc=exp_variance_pc,
        emulator=emulator,
    )
    return model, cfg


def _validate_shared_parameter_space(a: SystemModel, b: SystemModel) -> None:
    if a.param_names != b.param_names:
        raise ValueError(
            "Configs use different parameter order/names. "
            "Joint analysis requires identical parameter vectors."
        )

    if not np.allclose(a.prior.mins, b.prior.mins) or not np.allclose(
        a.prior.maxs, b.prior.maxs
    ):
        raise ValueError(
            "Configs use different prior bounds. "
            "Joint analysis requires identical prior ranges."
        )


def _default_output_file(a: SystemModel, b: SystemModel) -> Path:
    out_dir = Path("results") / "joint" / f"{a.name}__{b.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "chain.hdf5"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hic-bayes joint posterior for two systems (shared parameters)"
    )
    parser.add_argument("config_a", help="Path to first YAML config")
    parser.add_argument("config_b", help="Path to second YAML config")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output chain path (HDF5). Default: results/joint/<A>__<B>/chain.hdf5",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for emulator training and sampler workers.",
    )
    parser.add_argument(
        "--sampler",
        choices=["emcee", "ptemcee"],
        default=None,
        help="Override sampler from config A mcmc.sampler.",
    )
    args = parser.parse_args()

    # Build system models
    cfg_a_raw = _load_config(args.config_a)
    mcmc_cfg = cfg_a_raw.get("mcmc", {})
    n_jobs = args.n_jobs if args.n_jobs is not None else mcmc_cfg.get("n_threads", 1)

    print(f"Building model A from {args.config_a}")
    model_a, cfg_a = _build_system_model(args.config_a, n_jobs=n_jobs)

    print(f"Building model B from {args.config_b}")
    model_b, _ = _build_system_model(args.config_b, n_jobs=n_jobs)

    _validate_shared_parameter_space(model_a, model_b)

    print(
        f"\nJoint systems: {model_a.name} + {model_b.name}\n"
        f"Shared parameters: {len(model_a.param_names)}"
    )

    prior = model_a.prior
    n_params = prior.n_params
    sampler_name = args.sampler or mcmc_cfg.get("sampler", "emcee")
    n_walkers = mcmc_cfg.get("n_walkers", 128)
    n_steps = mcmc_cfg.get("n_steps", 2000)
    burnin = mcmc_cfg.get("burnin", n_steps // 2)
    thin = mcmc_cfg.get("thin", 1)
    n_temps = mcmc_cfg.get("n_temps", 10)
    seed = mcmc_cfg.get("seed", None)

    output_file = Path(args.output_file) if args.output_file else _default_output_file(model_a, model_b)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def _loglike_one(theta: np.ndarray, model: SystemModel) -> float:
        y_pred_pc, y_std_pc = model.emulator.predict(theta.reshape(1, -1), return_std=True)
        cov = build_covariance(y_std_pc.ravel() ** 2, model.exp_variance_pc)
        return mvn_loglike(model.y_exp_pc - y_pred_pc.ravel(), cov)

    def log_likelihood(theta: np.ndarray) -> float:
        ll_a = _loglike_one(theta, model_a)
        if not np.isfinite(ll_a):
            return -np.inf
        ll_b = _loglike_one(theta, model_b)
        if not np.isfinite(ll_b):
            return -np.inf
        return ll_a + ll_b

    def log_prob(theta: np.ndarray) -> float:
        lp = prior.log_prob(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    print(
        f"\nRunning {sampler_name}: walkers={n_walkers}, steps={n_steps}, "
        f"burnin={burnin}, thin={thin}, n_jobs={n_jobs}"
    )

    p0 = prior.sample(n_walkers, seed=seed)

    if sampler_name == "emcee":
        chain, log_prob_arr = run_emcee(
            log_prob,
            n_walkers,
            n_steps,
            n_params,
            p0,
            output_file=str(output_file),
            n_jobs=n_jobs,
            progress=True,
        )
    else:
        p0_pt = np.tile(p0, (n_temps, 1, 1))
        chain, log_prob_arr = run_ptemcee(
            log_likelihood,
            prior.log_prob,
            n_walkers,
            n_steps,
            n_params,
            p0_pt,
            n_temps=n_temps,
            output_file=str(output_file),
            n_jobs=n_jobs,
        )

    flat_chain = thin_chain(chain, burnin, thin)
    flat_log_prob = log_prob_arr[:, burnin::thin].ravel()

    summary = posterior_summary(flat_chain, model_a.param_names)
    print("\nPosterior summary (90% CI):")
    print(f"  {'Parameter':<25} {'Mean':>10} {'Std':>9} {'CI low':>10} {'CI high':>10}")
    print("  " + "-" * 70)
    for name, s in summary.items():
        print(
            f"  {name:<25} {s['mean']:>10.4f} {s['std']:>9.4f} "
            f"{s['ci_low']:>10.4f} {s['ci_high']:>10.4f}"
        )

    theta_map = map_estimate(flat_chain, flat_log_prob)
    print(f"\nMAP estimate: {dict(zip(model_a.param_names, theta_map))}")

    csv_path = output_file.with_suffix(".csv")
    pd.DataFrame(flat_chain, columns=model_a.param_names).to_csv(csv_path, index=False)

    corner_fig = corner_plot(
        flat_chain,
        model_a.param_labels,
        title=f"Joint posterior: {model_a.name} + {model_b.name}",
    )
    corner_path = output_file.parent / "posterior_corner_joint.pdf"
    corner_fig.savefig(corner_path, bbox_inches="tight")

    print(f"\nChain saved to:   {output_file}")
    print(f"Samples saved to: {csv_path}")
    print(f"Corner saved to:  {corner_path}")
    print("Done.")


if __name__ == "__main__":
    main()
