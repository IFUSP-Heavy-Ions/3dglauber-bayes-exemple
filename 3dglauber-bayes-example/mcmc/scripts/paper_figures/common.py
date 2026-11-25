"""Model registry, proposals and identical sample loading for every figure.

Sample policy (img/DESIGN_SYSTEM.md 1.2): first N_PROD production steps of the saved `chain`
dataset (burn-in already excluded), all walkers, step stride THIN. With 200 walkers this gives
200 * 20000 / 50 = 80 000 samples per model. Chains with 50 000 saved steps are truncated to
the first 20 000, which is what a 20 000-step run would have produced.
"""
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

import style

HERE = Path(__file__).resolve().parent
PROJECT = next(p for p in HERE.parents if (p / "production_designs").is_dir())
MCMC = PROJECT / "mcmc"
CACHE = MCMC / "04_paper_figures" / "cache"
DESIGN = (PROJECT / "production_designs/ProductionMaxPro-Mod/design_pts_Au_Au_200_production/"
          "design_ranges_main_AuAu-200.dat")

N_PROD = 20000
THIN = 50

LABELS = [
    r"$y_2$", r"$y_4$", r"$y_6$", r"$\sigma_{y_{\mathrm{loss}}}$",
    r"$\alpha_{\mathrm{rem}}$", r"$\alpha_{\mathrm{shadowing}}$",
    r"$\tau_{\mathrm{form}}$", r"$B_G\ [1/\mathrm{GeV}^2]$",
    r"String $\sigma_x$ [fm]", r"String $\sigma_\eta$",
    r"$\alpha_{\mathrm{shift}}$", r"$\frac{\eta}{s}\ T_{\mathrm{kink}}$",
    r"$\frac{\eta}{s}$ low-T slope", r"$\frac{\eta}{s}$ high-T slope",
    r"$\frac{\eta}{s}$ at kink", r"$\frac{\zeta}{s}$ max",
    r"$\frac{\zeta}{s}\ T_{\mathrm{peak}}$", r"$\frac{\zeta}{s}$ width",
    r"$\frac{\zeta}{s}\ \lambda$ asym.", r"$e_{\mathrm{switch}}$",
]

MODELS = {
    "baseline": dict(path=MCMC / "01_01_baseline_reproduction/chain-idf-0.hdf",
                     label="baseline (diag)", color=style.BASELINE, ls="-"),
    "cs_global": dict(path=MCMC / "02_01_global_cs/chain-idf-0-cs-system-rho-1.hdf",
                      label="CS global", color=style.CS, ls=style.DASH),
    "rbf_global": dict(path=MCMC / "02_02_global_rbf/chain-idf-0-rbf-global-c33aba121315.hdf",
                       label="RBF global", color=style.RBF, ls=style.DASH),
    "cs_sector": dict(path=MCMC / "02_04_cs_sector/chain-idf-0-cs-observable-region-rho-1-a77902df73a5.hdf",
                      label="CS sector", color=style.CS, ls=style.DASH),
    "rbf_sector": dict(path=MCMC / "02_05_rbf_sector/chain-idf-0-rbf-observable-a873ae558cc9.hdf",
                       label="RBF sector", color=style.RBF, ls=style.DASH),
    # Proposal 4 (covmom): theta + Gamma-skewed nuisances on STAR v2(eta) 20-70% (skews from Tensor PCA of replicas).
    # focus=True: not used for the shared y limits of the 40-panel F05 pages.
    "nuis_global": dict(path=MCMC / "02_10_nongaussian_nuisance/chain-idf-0-nongauss-nuisance-6058bef13723.hdf",
                        label="skew nuisance (global norm.)", color=style.EXTRA, ls=style.DASHDOT, focus=True),
    "nuis_split": dict(path=MCMC / "02_11_nongaussian_nuisance_detector_split/chain-idf-0-nongauss-nuisance-detector-split.hdf",
                       label="skew nuisance (TPC/FTPC norm.)", color=style.OLIVE, ls=(0, (1, 1.5)), focus=True),
    # Proposal 3 (covinf): PHOBOS dN/deta covariance learned from cross-detector data (fatia 2a); job 5079.
    # focus=True keeps it out of the shared F05 y limits (it is not one of the proposal 1/2 models).
    "learned_phobos": dict(path=MCMC / "02_14_learned_phobos_normalization/chain-idf-0-learned-phobos-7ec3735f96a8.hdf",
                           label="learned PHOBOS cov. (data)", color=style.LEARNED, ls=style.DASH, focus=True),
}

PROPOSALS = {
    "global": ["baseline", "cs_global", "rbf_global"],
    "sector": ["baseline", "cs_sector", "rbf_sector"],
    "covinf": ["baseline", "cs_sector", "learned_phobos"],
    "covmom": ["baseline", "cs_sector", "nuis_global", "nuis_split"],
}

SHORT = {"cs_global": "cs", "rbf_global": "rbf", "cs_sector": "cs", "rbf_sector": "rbf",
         "nuis_global": "nuisglob", "nuis_split": "nuissplit", "learned_phobos": "learnphob"}


def design():
    with DESIGN.open() as f:
        rows = list(csv.DictReader(f))
    names = [r["param"] for r in rows]
    lo = np.array([float(r["min"]) for r in rows])
    hi = np.array([float(r["max"]) for r in rows])
    return names, lo, hi


def prior_ranges():
    """Table I ranges for axes: y4 in [0,4], y6 in [0,6] (design min ~0)."""
    _, lo, hi = design()
    lo = lo.copy()
    lo[:6] = 0.0
    lo[10] = 0.0
    return list(zip(lo, hi))


def available(model):
    return MODELS[model]["path"].is_file()


def load(model):
    """(n_samples, 20) array, identical policy for every model; cached as .npz."""
    path = MODELS[model]["path"]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{model}_prod{N_PROD}_thin{THIN}.npz"
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}|{N_PROD}|{THIN}"
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return z["samples"]
    with h5py.File(path, "r") as f:
        d = f["chain"]
        walkers, steps, ndim = d.shape
        if steps < N_PROD:
            raise ValueError(f"{model}: {steps} saved production steps < {N_PROD}")
        blocks = [d[:, s:min(s + 2000, N_PROD):THIN, :20] for s in range(0, N_PROD, 2000)]
    samples = np.concatenate(blocks, axis=1).reshape(-1, 20)
    if not np.isfinite(samples).all():
        raise ValueError(f"{model}: non-finite samples")
    np.savez_compressed(cache, samples=samples, key=key)
    print(f"{model}: {walkers} walkers x {N_PROD} steps / {THIN} -> {len(samples)} samples", flush=True)
    return samples


def provenance(models):
    """Dict written next to each figure: chains, sample policy, script hashes."""
    out = dict(n_prod=N_PROD, thin=THIN, credible_level=style.CI, seed=style.SEED, chains={})
    for m in models:
        out["chains"][m] = str(MODELS[m]["path"].relative_to(PROJECT))
    out["scripts"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                      for p in sorted(HERE.glob("*.py"))}
    return out


def write_manifest(outdir, stem, data):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{stem}.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
