#!/usr/bin/env python3
"""MCMC with GLOBAL RBF experimental correlations in common physical coordinates (pT, eta, centrality).

Same sampler, priors, emulators and likelihood as the CS-global / sector studies; only the
experimental covariance changes (experimental_global_covariance.py). Cross-system covariance is zero.

  python bayes_mcmc_rbf_global.py --rbf-length-pt 0.5 --rbf-length-eta 1 --rbf-length-centrality 10 --check-covariance
  python bayes_mcmc_rbf_global.py --rbf-length-pt 0.5 --rbf-length-eta 1 --rbf-length-centrality 10 \
      --nwalkers 200 --nburnsteps 20000 --ncpus 20 --status 1000 20000
"""
import argparse
import json
from pathlib import Path

import numpy as np
from configurations import do_transform_design, transform_design  # noqa: F401  (legacy emulator namespace)

from experimental_global_covariance import (AXES, AXIS_UNITS, DEFAULT_ACCEPTANCE, global_rbf_covariance,
                                            load_acceptance, system_coordinates)
from experimental_sector_covariance import covariance_signature, likelihood_std


def check_covariance(lengths, acceptance):
    """Build both system matrices from the experimental files only (no emulator, no MCMC)."""
    import pandas as pd
    import configurations as config
    report, matrices, slices = [], {}, {}
    for system in config.system_strs:
        slices[system], sigmas, offset = [], [], 0
        for obs in config.active_obs_list[system]:
            edges = config.obs_cent_list[system][obs]
            n = len(edges)
            f = Path(config.dir_obs_exp) / system / config.expt_label_dict[obs[-4:]] / (obs + ".dat")
            raw = pd.read_csv(f, sep=" ", skiprows=2, escapechar="#")
            mean = raw["val"].iloc[:n].to_numpy(float)
            std = likelihood_std(obs, mean, np.sqrt(raw["err"].iloc[:n].to_numpy(float)), config.transform_multiplicities)
            slices[system].append((obs, slice(offset, offset + n)))
            sigmas.extend(std)
            offset += n
        m = global_rbf_covariance(system, slices[system], np.array(sigmas), config.obs_cent_list, lengths, acceptance)
        eig = np.linalg.eigvalsh(m)
        tol = len(eig) * np.finfo(float).eps * max(abs(eig).max(), np.finfo(float).tiny)
        if eig.min() < -tol:
            raise ValueError(f"Covariance is not positive semidefinite: {system}")
        np.testing.assert_allclose(np.diag(m), np.square(sigmas), rtol=1e-14, atol=0)
        d = np.sqrt(np.diag(m))
        r = m / np.outer(d, d)
        off = r[~np.eye(len(r), dtype=bool)]
        x = system_coordinates(system, slices[system], config.obs_cent_list, acceptance)
        matrices[system] = m
        report.append(dict(system=system, n_bins=len(sigmas), numerical_rank=int((eig > tol).sum()),
                           min_eigenvalue=float(eig.min()), psd_tolerance=float(tol),
                           offdiag_correlation=dict(mean=float(off.mean()), median=float(np.median(off)),
                                                    frac_above_0p5=float((off > 0.5).mean())),
                           coordinates={obs: np.round(x[s][[0, -1]], 4).tolist() for obs, s in slices[system]}))
    return dict(model="rbf_global", lengths=lengths, signature=covariance_signature(matrices, slices, config.obs_cent_list),
                systems=report)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nsteps", type=int, nargs="?", help="production steps")
    ap.add_argument("--nwalkers", type=int)
    ap.add_argument("--nburnsteps", type=int)
    ap.add_argument("--ntemps", type=int, default=1)
    ap.add_argument("--status", type=int)
    ap.add_argument("--ncpus", type=int, default=20)
    ap.add_argument("--chain-path", type=Path)
    ap.add_argument("--acceptance-json", type=Path, default=DEFAULT_ACCEPTANCE)
    ap.add_argument("--check-covariance", action="store_true")
    for axis in AXES:
        ap.add_argument(f"--rbf-length-{axis}", type=float, required=True, help=f"length in {AXIS_UNITS[axis]}")
    a = ap.parse_args()
    lengths = {axis: getattr(a, f"rbf_length_{axis}") for axis in AXES}
    acceptance = load_acceptance(a.acceptance_json)
    if a.check_covariance:
        print(json.dumps(check_covariance(lengths, acceptance), indent=2))
        return
    if a.nsteps is None or a.nsteps <= 0:
        ap.error("Specify a positive nsteps or --check-covariance.")
    from bayes_mcmc_rbf_global_chain import Chain
    from emulator import _Covariance
    globals()["_Covariance"] = _Covariance  # emulators pickled from emulator.py's __main__
    kwargs = dict(nsteps=a.nsteps, nwalkers=a.nwalkers, nburnsteps=a.nburnsteps, ntemps=a.ntemps,
                  status=a.status, ncpu=a.ncpus)
    Chain(lengths, acceptance, a.acceptance_json.resolve(), path=a.chain_path).run_mcmc(**kwargs)


if __name__ == "__main__":
    main()
