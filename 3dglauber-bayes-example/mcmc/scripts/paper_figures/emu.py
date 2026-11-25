"""Emulator access for F05/F26, using the validated path of mcmc/scripts/build_sector_studies.py.

Loads emulator/emulator-<system>-idf-0-npc-10.dill, rebinds to the current Emulator class
(diagnose_phase2_numerical.load_emulator_module) and predicts in physical units
(dN/deta and dET/deta are exp-transformed back). Predictions are cached per model.
"""
import csv
import sys

import dill
import numpy as np

import common
import style

sys.path.insert(0, str(common.PROJECT))
sys.path.insert(0, str(common.MCMC / "scripts"))
BIN_INDEX = common.MCMC / "02_08_sector_input_correlations/bin_index.csv"
N_PRED = 10000
_EMU = {}


def _module():
    from diagnose_phase2_numerical import load_emulator_module
    return load_emulator_module()


def emulators():
    if not _EMU:
        m = _module()
        cfg = __import__("configurations")
        for system in cfg.system_strs:
            with (common.PROJECT / f"emulator/emulator-{system}-idf-0-npc-10.dill").open("rb") as h:
                e = dill.load(h)
            e.__class__ = m.Emulator
            _EMU[system] = (e, list(cfg.active_obs_list[system]))
    return _EMU


def physical(obs, v):
    return np.exp(v) if ("dN" in obs or "dET" in obs) else v


def bins():
    with BIN_INDEX.open() as f:
        return list(csv.DictReader(f))


def theta_subset(model):
    s = common.load(model)
    rng = np.random.default_rng(style.SEED)
    return s[rng.choice(len(s), N_PRED, replace=False)]


def predictions(model):
    """{system: {obs: (N_PRED, nbins)}} in physical units, cached."""
    cache = common.CACHE / f"pred_{model}_n{N_PRED}_seed{style.SEED}.npz"
    if cache.is_file():
        z = np.load(cache)
        out = {}
        for k in z.files:
            system, obs = k.split("|")
            out.setdefault(system, {})[obs] = z[k]
        return out
    theta = theta_subset(model)
    out, flat = {}, {}
    for system, (e, observables) in emulators().items():
        mean = {}
        for start in range(0, len(theta), 2000):
            part = e.predict(theta[start:start + 2000], return_cov=False)
            for obs in observables:
                mean.setdefault(obs, []).append(np.asarray(part[obs]))
        out[system] = {obs: physical(obs, np.concatenate(v)) for obs, v in mean.items()}
        flat.update({f"{system}|{obs}": v for obs, v in out[system].items()})
    np.savez_compressed(cache, **flat)
    print(f"{model}: emulator predictions for {len(theta)} samples cached", flush=True)
    return out


def emulator_correlation(system, obs, n_points=60):
    """Emulator-uncertainty correlation of one observable block, averaged over
    n_points baseline-posterior samples (same sampling as 02_12/bin_level.npz)."""
    e, _ = emulators()[system]
    rng = np.random.default_rng(style.SEED)
    base = common.load("baseline")
    theta = base[rng.choice(len(base), n_points, replace=False)]
    _, cov = e.predict(theta, return_cov=True, extra_std=np.zeros(len(theta)))
    c = np.asarray(cov[obs, obs]).mean(axis=0)
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)
