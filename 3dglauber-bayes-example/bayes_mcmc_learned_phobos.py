#!/usr/bin/env python3
"""MCMC with the experimental covariance of PHOBOS dN/deta LEARNED from cross-detector data (degrau 3).

Same sampler, priors, emulators and likelihood as the baseline / CS / RBF studies. Only the Au-Au
PHOBOS dN/deta block (594 bins) gets off-diagonal terms (experimental_learned_covariance.py);
all other bins and the d-Au system stay diagonal. Diagonal unchanged: only correlation is learned.

  python bayes_mcmc_learned_phobos.py --check-covariance
  python bayes_mcmc_learned_phobos.py --nwalkers 200 --nburnsteps 20000 --ncpus 20 --status 1000 20000
"""
import argparse
import json
from pathlib import Path

import numpy as np
from configurations import do_transform_design, transform_design  # noqa: F401  (legacy emulator namespace)

from experimental_learned_covariance import DEFAULTS, learned_phobos_covariance, summary
from experimental_sector_covariance import covariance_signature, likelihood_std


def check_covariance(u, slope):
    """Build both system matrices from the experimental files only (no emulator, no MCMC)."""
    import pandas as pd
    import configurations as config
    report, matrices, slices = [], {}, {}
    for system in config.system_strs:
        slices[system], sigmas, offset = [], [], 0
        for obs in config.active_obs_list[system]:
            n = len(config.obs_cent_list[system][obs])
            f = Path(config.dir_obs_exp) / system / config.expt_label_dict[obs[-4:]] / (obs + ".dat")
            raw = pd.read_csv(f, sep=" ", skiprows=2, escapechar="#")
            mean = raw["val"].iloc[:n].to_numpy(float)
            std = likelihood_std(obs, mean, np.sqrt(raw["err"].iloc[:n].to_numpy(float)), config.transform_multiplicities)
            slices[system].append((obs, slice(offset, offset + n)))
            sigmas.extend(std)
            offset += n
        m = learned_phobos_covariance(system, slices[system], np.array(sigmas), u=u, slope=slope)
        matrices[system] = m
        report.append(dict(summary(system, slices[system], np.array(sigmas), m), n_bins=len(sigmas)))
    return dict(model="learned_phobos", u=u, slope=slope,
                signature=covariance_signature(matrices, slices, config.obs_cent_list), systems=report)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nsteps", type=int, nargs="?", help="production steps")
    ap.add_argument("--nwalkers", type=int)
    ap.add_argument("--nburnsteps", type=int)
    ap.add_argument("--ntemps", type=int, default=1)
    ap.add_argument("--status", type=int)
    ap.add_argument("--ncpus", type=int, default=20)
    ap.add_argument("--chain-path", type=Path)
    ap.add_argument("--u", type=float, default=DEFAULTS["u"], help="uncorrelated class-to-class part (log units)")
    ap.add_argument("--slope", type=float, default=DEFAULTS["slope"], help="centrality slope rms per 100%% centrality")
    ap.add_argument("--check-covariance", action="store_true")
    a = ap.parse_args()
    if a.check_covariance:
        print(json.dumps(check_covariance(a.u, a.slope), indent=2))
        return
    if a.nsteps is None or a.nsteps <= 0:
        ap.error("Specify a positive nsteps or --check-covariance.")
    from bayes_mcmc_learned_phobos_chain import Chain
    from emulator import _Covariance
    globals()["_Covariance"] = _Covariance  # emulators pickled from emulator.py's __main__
    kwargs = dict(nsteps=a.nsteps, nwalkers=a.nwalkers, nburnsteps=a.nburnsteps, ntemps=a.ntemps,
                  status=a.status, ncpu=a.ncpus)
    Chain(a.u, a.slope, path=a.chain_path).run_mcmc(**kwargs)


if __name__ == "__main__":
    main()
