#!/usr/bin/env python3
"""Posterior shift (sector vs baseline) vs emulator/experimental variance ratio, per parameter.

Both covariances in the likelihood live in OBSERVABLE space (bins), not parameter space:
    log L = -1/2 dY^T (Sigma_exp + Sigma_emu(theta))^{-1} dY
so a per-parameter ratio needs a map from bins to parameters. We use the emulator
Jacobian J_ij = d mu_i / d theta_j (same transformed coordinates as the likelihood),
scaled by the baseline posterior width sigma_j, and weight each bin by the Fisher
information it carries about theta_j:

    w_ij = (J_ij sigma_j)^2 / (sigma_exp,i^2 + sigma_emu,i^2)   (information the likelihood
                                                                actually gets from bin i)
    r_i  = sigma_emu,i^2 / sigma_exp,i^2                       (diagonal ratio per bin)
    R_j  = sum_i w_ij r_i / sum_i w_ij                         (ratio "seen" by theta_j)
    f_j  = sum_{i in sector bins} w_ij / sum_i w_ij            (share of the info that comes
                                                                from bins whose covariance the
                                                                CS / RBF sector modifies)
Also R_j restricted to sector bins. The CS / RBF sector models keep the diagonal of
Sigma_exp and only add correlations inside the rapidity regions, so r_i is the same for
baseline, CS and RBF. Everything is averaged over N baseline posterior samples.

Shifts, as in the AGORA 2026 presentation: (median_model - median_baseline) / sigma_B, with
sigma_B the standard deviation of the baseline posterior.
  - sector CS / RBF (test 2): computed here from the chains in mcmc/;
  - global CS / RBF (test 1, Au-Au only, hic-bayes framework): read from
    run_001_cs_rbf_median_shifts_baseline_sigma.csv. Global correlations act on all Au-Au
    bins, so their ratio is R_j restricted to Au-Au bins. That run used another emulator file
    (same design and bins); here the Jacobian of this repository's emulator is a proxy.
R_j does not depend on the scale sigma_j (it cancels in the weighted mean).

Usage (from 3dglauber-bayes-example, on the server):
    python mcmc/scripts/shift_vs_cov_ratio.py [--nsamples 60]
Output: mcmc/02_12_shift_vs_cov_ratio/
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configurations import do_transform_design, transform_design  # noqa: F401,E402  (dill namespace)

OUT = ROOT / "mcmc/02_12_shift_vs_cov_ratio"
BASELINE = ROOT / "mcmc/01_01_baseline_reproduction/chain-idf-0.hdf"
SECTORS = {"cs": ROOT / "mcmc/02_04_cs_sector/chain-idf-0-cs-observable-region-rho-1-a77902df73a5.hdf",
           "rbf": ROOT / "mcmc/02_05_rbf_sector/chain-idf-0-rbf-observable-a873ae558cc9.hdf"}
REGIONS = json.loads((ROOT / "rapidity_regions.json").read_text())
GLOBAL_SHIFTS = Path("/storage/avilamrs/results/hic-bayes/AllDetectorsAuAu200/uncertainty_correlation/analysis/"
                     "plots/run_004/presentation_run_001_marginals/run_001_cs_rbf_median_shifts_baseline_sigma.csv")


def load_samples(path, stride=50, max_samples=200000):
    with h5py.File(path, "r") as f:
        d = f["chain"]; w, n, k = d.shape
        stride = max(stride, int(np.ceil(w * n / max_samples)))
        return np.concatenate([d[:, s:min(s + 256 * stride, n):stride, :].reshape(-1, k)
                               for s in range(0, n, 256 * stride)])[:, :20]


def stats_of(x):
    q16, q50, q84 = np.quantile(x, [.16, .5, .84], axis=0)
    return dict(median=q50, s68=(q84 - q16) / 2, mean=x.mean(0), std=x.std(0))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsamples", type=int, default=60)
    ap.add_argument("--rel-step", type=float, default=0.01, help="finite-difference step / prior range")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import bayes_mcmc
    from emulator import _Covariance
    globals()["_Covariance"] = _Covariance
    chain = bayes_mcmc.Chain(path=OUT / "unused.hdf")
    labels = [str(l) for l in chain.labels][:20] if len(chain.labels) >= 20 else None
    design = list(csv.DictReader(open(ROOT / "mcmc/02_06_baseline_vs_cs_sector/posterior_parameter_comparison.csv")))
    names = [r["parameter"] for r in design]
    lo, hi = np.asarray(chain.min[:20], float), np.asarray(chain.max[:20], float)

    base = load_samples(BASELINE)
    bst = stats_of(base)
    shifts = {}
    for key, path in SECTORS.items():
        sst = stats_of(load_samples(path))
        shifts[key] = dict(median_over_sigmaB=(sst["median"] - bst["median"]) / bst["std"],
                           median_over_s68=(sst["median"] - bst["median"]) / bst["s68"],
                           mean_over_std=(sst["mean"] - bst["mean"]) / bst["std"],
                           width_ratio=sst["s68"] / bst["s68"])

    # bins: experimental variance, sector membership
    sig_exp2, sector, obs_of, auau = [], [], [], []
    for sysname in chain._expt_y:
        sig_exp2.append(np.diag(chain._expt_cov[sysname]))
        m = np.zeros(chain._expt_y[sysname].size, bool)
        for obs, slc in chain._slices[sysname]:
            m[slc] = obs in REGIONS.get(sysname, {})
            obs_of += [f"{sysname}/{obs}"] * (slc.stop - slc.start)
        sector.append(m); auau.append(np.full(m.size, sysname == "Au-Au-200"))
    sig_exp2, sector, auau = np.concatenate(sig_exp2), np.concatenate(sector), np.concatenate(auau)

    def predict(X):
        """mean (n, nbins) and emulator variance (n, nbins), concatenated over systems."""
        pred = chain._predict(X, return_cov=True)
        mus, vs = [], []
        for sysname in chain._expt_y:
            Y, C = pred[sysname]
            mus.append(np.concatenate([Y[o] for o, _ in chain._slices[sysname]], axis=1))
            vs.append(np.concatenate([np.diagonal(C[o, o], axis1=1, axis2=2) for o, _ in chain._slices[sysname]], axis=1))
        return np.concatenate(mus, 1), np.concatenate(vs, 1)

    rng = np.random.default_rng(args.seed)
    pts = base[rng.choice(len(base), args.nsamples, replace=False)]
    h = args.rel_step * (hi - lo)
    W = np.zeros((20, sig_exp2.size)); W_exp = np.zeros_like(W); r_acc = np.zeros(sig_exp2.size)
    for t in pts:
        X = np.repeat(t[None], 41, 0)
        for j in range(20):
            X[1 + 2 * j, j] = min(t[j] + h[j], hi[j]); X[2 + 2 * j, j] = max(t[j] - h[j], lo[j])
        mu, var = predict(X)
        J = np.stack([(mu[1 + 2 * j] - mu[2 + 2 * j]) / (X[1 + 2 * j, j] - X[2 + 2 * j, j]) for j in range(20)])
        S2 = (J * bst["s68"][:, None]) ** 2
        W += S2 / (sig_exp2 + var[0])[None, :]
        W_exp += S2 / sig_exp2[None, :]
        r_acc += var[0] / sig_exp2
    W /= len(pts); W_exp /= len(pts); r = r_acc / len(pts)
    info_kept = W.sum(1) / W_exp.sum(1)        # 1/(1+r) averaged with exp-only weights

    R = (W * r).sum(1) / W.sum(1)
    R_sec = (W[:, sector] * r[sector]).sum(1) / W[:, sector].sum(1)
    f_sec = W[:, sector].sum(1) / W.sum(1)
    R_auau = (W[:, auau] * r[auau]).sum(1) / W[:, auau].sum(1)
    g = {row["parameter"]: row for row in csv.DictReader(open(GLOBAL_SHIFTS))}
    shifts["cs_global"] = dict(median_over_sigmaB=np.array([float(g[n]["cs_shift_in_baseline_sigma"]) for n in names]))
    shifts["rbf_global"] = dict(median_over_sigmaB=np.array([float(g[n]["rbf_shift_in_baseline_sigma"]) for n in names]))
    top_obs = []
    for j in range(20):
        agg = {}
        for i, o in enumerate(obs_of):
            agg[o] = agg.get(o, 0) + W[j, i]
        o, v = max(agg.items(), key=lambda kv: kv[1]); top_obs.append(f"{o} ({v / W[j].sum():.0%})")

    rows = []
    for j in range(20):
        rows.append(dict(parameter=names[j],
                         shift_cs_global_sigmaB=shifts["cs_global"]["median_over_sigmaB"][j],
                         shift_rbf_global_sigmaB=shifts["rbf_global"]["median_over_sigmaB"][j],
                         shift_cs_sector_sigmaB=shifts["cs"]["median_over_sigmaB"][j],
                         shift_rbf_sector_sigmaB=shifts["rbf"]["median_over_sigmaB"][j],
                         shift_cs_median_s68=shifts["cs"]["median_over_s68"][j],
                         shift_cs_mean_std=shifts["cs"]["mean_over_std"][j],
                         shift_rbf_median_s68=shifts["rbf"]["median_over_s68"][j],
                         shift_rbf_mean_std=shifts["rbf"]["mean_over_std"][j],
                         width_ratio_cs=shifts["cs"]["width_ratio"][j], width_ratio_rbf=shifts["rbf"]["width_ratio"][j],
                         R_emu_over_exp=R[j], R_emu_over_exp_AuAu_bins=R_auau[j], R_emu_over_exp_sector_bins=R_sec[j], info_share_sector_bins=f_sec[j],
                         info_kept_with_emulator=info_kept[j], fisher_info_effective=W[j].sum(),
                         dominant_observable=top_obs[j]))
    with open(OUT / "shift_vs_cov_ratio.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    np.savez_compressed(OUT / "bin_level.npz", r=r, sig_exp2=sig_exp2, sector=sector, W=W, W_exp=W_exp,
                        obs=np.array(obs_of), names=np.array(names))

    tests = [("cs_global", "CS global (test 1)", R_auau, r"$R_j$ (Au-Au bins)", "#477fb5"),
             ("rbf_global", "RBF global (test 1)", R_auau, r"$R_j$ (Au-Au bins)", "#d17a22"),
             ("cs", "CS sector (test 2)", R_sec, r"$R_j$ (sector bins)", "#477fb5"),
             ("rbf", "RBF sector (test 2)", R_sec, r"$R_j$ (sector bins)", "#d17a22")]
    corr = {}
    for key, *_ in tests:
        s = np.abs(shifts[key]["median_over_sigmaB"])
        corr[key] = {m: spearman(s, v) for m, v in (("R_all", R), ("R_AuAu", R_auau), ("R_sector", R_sec),
                                                     ("info_share_sector", f_sec), ("info_kept", info_kept))}

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    short = [n.replace("bulk_viscosity_3_zeta_over_s_", "zeta_").replace("shear_viscosity_3_eta_over_s_", "eta_")
             .replace("_in_GeV", "") for n in names]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for col, (key, title, xr, xl, color) in enumerate(tests):
        s = np.abs(shifts[key]["median_over_sigmaB"])
        for row, (x, lab, mk) in enumerate(((xr, xl, "R_AuAu" if "global" in key else "R_sector"),
                                             (R, r"$R_j$ (all 917 bins)", "R_all"))):
            ax = axes[row, col]
            ax.scatter(x, s, color=color, s=30)
            for j in np.argsort(-s)[:5]:
                ax.annotate(short[j], (x[j], s[j]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("log")
            ax.set(xlabel=lab + r"  $=\langle\sigma^2_\mathrm{emu}/\sigma^2_\mathrm{exp}\rangle_{w_j}$",
                   ylabel=r"|median shift| ($\sigma_B$)")
            ax.set_title(f"{title}: Spearman = {corr[key][mk]:+.2f}", fontsize=10)
    fig.suptitle("Posterior median shift vs emulator/experimental variance ratio seen by each parameter")
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(OUT / "shift_vs_cov_ratio.png", dpi=150); fig.savefig(OUT / "shift_vs_cov_ratio.pdf")

    # bin-level ratio, for reference
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.semilogy(r, ".", ms=3, color="gray", label="all bins")
    ax.semilogy(np.flatnonzero(sector), r[sector], ".", ms=4, color="#c0392b", label="sector bins (CS/RBF modify)")
    ax.axhline(1, color="k", lw=.7); ax.set(xlabel="bin index (Au-Au then d-Au)", ylabel=r"$\sigma^2_\mathrm{emu}/\sigma^2_\mathrm{exp}$")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "bin_ratio_emu_over_exp.png", dpi=150)

    (OUT / "summary.json").write_text(json.dumps(dict(
        nsamples=args.nsamples, rel_step=args.rel_step, spearman_abs_shift_mean_std=corr,
        bins=int(r.size), sector_bins=int(sector.sum()),
        median_bin_ratio_all=float(np.median(r)), median_bin_ratio_sector=float(np.median(r[sector]))), indent=1))
    order = np.argsort(-np.abs(shifts["cs"]["median_over_sigmaB"]))
    print(f"{'parameter':42s} {'CSg':>6s} {'RBFg':>6s} {'CSs':>6s} {'RBFs':>6s} {'R_all':>7s} {'R_AuAu':>7s} {'R_sec':>6s} {'f_sec':>5s}")
    for j in order:
        print(f"{names[j][:42]:42s} " + " ".join(f"{shifts[k]['median_over_sigmaB'][j]:+6.3f}" for k in ("cs_global", "rbf_global", "cs", "rbf"))
              + f" {R[j]:7.2f} {R_auau[j]:7.2f} {R_sec[j]:6.2f} {f_sec[j]:5.2f}")
    print(json.dumps(corr, indent=1))


if __name__ == "__main__":
    main()
