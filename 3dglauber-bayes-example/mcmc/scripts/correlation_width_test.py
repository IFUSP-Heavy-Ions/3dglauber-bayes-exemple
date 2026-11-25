#!/usr/bin/env python3
"""Do experimental correlations widen the posterior when experimental errors are large?

Hypothesis (advisor): with wide experimental error bars, adding correlations to Sigma_exp
widens the posterior. Test in the linear-Gaussian approximation of the likelihood
used by the chains (emulator mean linearized around posterior points):

    Cov_m = ( sum_sys J^T (Sigma_exp,m + Sigma_emu(theta))^-1 J  +  P )^-1
    P = diag(12 / range^2)        (Gaussian stand-in for the uniform prior)

for m = baseline (diagonal), CS sector (rho=1 within system+observable+rapidity region) and
RBF sector (lengths: centrality 10, eta 1, pT 0.5 GeV), exactly as built by
experimental_sector_covariance.sector_covariance. J and Sigma_emu are averaged over N
baseline posterior points (F is averaged, then inverted).

1. Validation: predicted width ratios sqrt(Cov_m,jj / Cov_base,jj) vs the MCMC std ratios.
2. Hypothesis scan: Sigma_exp -> k^2 Sigma_exp (k = 0.25 ... 8) for baseline and sector
   alike; width ratio vs k per parameter.
3. Mechanism: whiten J by sigma_exp; inside each correlated sector split it into the common
   (sector-mean) component and the shape component. c_j = common-mode share of the
   information about theta_j. Positive correlation removes information from the common
   mode and adds it to the shape, so the sign of the effect is expected to follow c_j.

Usage (from 3dglauber-bayes-example, on the server):
    python mcmc/scripts/correlation_width_test.py [--nsamples 40]
Output: mcmc/02_13_correlation_width_test/
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

OUT = ROOT / "mcmc/02_13_correlation_width_test"
BASELINE = ROOT / "mcmc/01_01_baseline_reproduction/chain-idf-0.hdf"
CHAINS = {"cs": ROOT / "mcmc/02_04_cs_sector/chain-idf-0-cs-observable-region-rho-1-a77902df73a5.hdf",
          "rbf": ROOT / "mcmc/02_05_rbf_sector/chain-idf-0-rbf-observable-a873ae558cc9.hdf"}
RBF_LENGTHS = {"centrality": 10.0, "eta": 1.0, "pt": 0.5}
KS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RHOS = [0.3, 0.6, 0.9, 0.99]


def load_samples(path, stride=50, max_samples=200000):
    with h5py.File(path, "r") as f:
        d = f["chain"]; w, n, k = d.shape
        stride = max(stride, int(np.ceil(w * n / max_samples)))
        return np.concatenate([d[:, s:min(s + 256 * stride, n):stride, :].reshape(-1, k)
                               for s in range(0, n, 256 * stride)])[:, :20]


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plt_scan(scan, rows, names, c_common):
    models = list(scan)
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8), sharey=True)
    for ax, m in zip(axes, models):
        for j in range(20):
            ax.plot(KS, [rows[j][f"{m}_k{k:g}"] for k in KS], color=plt.cm.viridis(c_common[j]), lw=1)
        ax.axhline(1, color="k", lw=.8); ax.axvline(1, color="gray", ls=":", lw=.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(m.replace("cs_rho", "CS sector rho=").replace("cs", "CS sector rho=1") if m != "rbf" else "RBF sector", fontsize=10)
        ax.set_xlabel(r"experimental error scale $k$")
    axes[0].set_ylabel("width ratio sector / baseline (linear)")
    fig.suptitle("Error-scale scan: correlations widen (ratio > 1) or narrow (< 1) the posterior")
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(OUT / "rho_k_scan.png", dpi=150); fig.savefig(OUT / "rho_k_scan.pdf")
    return fig, axes


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsamples", type=int, default=40)
    ap.add_argument("--rel-step", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import bayes_mcmc
    from emulator import _Covariance
    from configurations import obs_cent_list
    from experimental_sector_covariance import sector_covariance, rapidity_region_labels
    globals()["_Covariance"] = _Covariance
    regions = json.loads((ROOT / "rapidity_regions.json").read_text())
    chain = bayes_mcmc.Chain(path=OUT / "unused.hdf")
    systems = list(chain._expt_y)
    names = [r["parameter"] for r in csv.DictReader(open(ROOT / "mcmc/02_06_baseline_vs_cs_sector/posterior_parameter_comparison.csv"))]
    lo, hi = np.asarray(chain.min[:20], float), np.asarray(chain.max[:20], float)
    P = np.diag(12.0 / (hi - lo) ** 2)

    # experimental covariances (likelihood coordinates)
    std = {s: np.sqrt(np.diag(chain._expt_cov[s])) for s in systems}
    Sexp = {"baseline": {s: np.diag(std[s] ** 2) for s in systems},
            "cs": {s: sector_covariance(s, chain._slices[s], std[s], obs_cent_list, "cs", rho=1.0, regions=regions) for s in systems},
            "rbf": {s: sector_covariance(s, chain._slices[s], std[s], obs_cent_list, "rbf", lengths=RBF_LENGTHS, regions=regions) for s in systems}}
    for rho in RHOS:   # CS sector with rho < 1 (rho = 1 makes each block rank one)
        Sexp[f"cs_rho{rho:g}"] = {s: sector_covariance(s, chain._slices[s], std[s], obs_cent_list, "cs", rho=rho, regions=regions) for s in systems}
    # sector membership (same blocks for CS and RBF)
    blocks = {s: [] for s in systems}
    for s in systems:
        idx = np.arange(std[s].size)
        for obs, slc in chain._slices[s]:
            labels = rapidity_region_labels(s, obs, obs_cent_list[s][obs], regions)
            for lab in np.unique(labels):
                blocks[s].append(idx[slc][np.flatnonzero(labels == lab)])

    def predict(X):
        pred = chain._predict(X, return_cov=True)
        mu, cov = {}, {}
        for s in systems:
            Y, C = pred[s]
            sl = chain._slices[s]; n = std[s].size
            mu[s] = np.concatenate([Y[o] for o, _ in sl], axis=1)
            M = np.empty((X.shape[0], n, n))
            for o1, s1 in sl:
                for o2, s2 in sl:
                    M[:, s1, s2] = C[o1, o2]
            cov[s] = M
        return mu, cov

    base = load_samples(BASELINE)
    rng = np.random.default_rng(args.seed)
    pts = base[rng.choice(len(base), args.nsamples, replace=False)]
    h = args.rel_step * (hi - lo)
    F = {(m, k): np.zeros((20, 20)) for m in Sexp for k in KS}
    F_exp_only = {m: np.zeros((20, 20)) for m in Sexp}
    common = np.zeros(20); total = np.zeros(20)
    for t in pts:
        X = np.repeat(t[None], 41, 0)
        for j in range(20):
            X[1 + 2 * j, j] = min(t[j] + h[j], hi[j]); X[2 + 2 * j, j] = max(t[j] - h[j], lo[j])
        mu, cov = predict(X)
        for s in systems:
            J = np.stack([(mu[s][1 + 2 * j] - mu[s][2 + 2 * j]) / (X[1 + 2 * j, j] - X[2 + 2 * j, j]) for j in range(20)], axis=1)
            Semu = cov[s][0]
            for m in Sexp:
                for k in KS:
                    F[m, k] += J.T @ np.linalg.solve(k * k * Sexp[m][s] + Semu, J)
                F_exp_only[m] += J.T @ np.linalg.solve(Sexp[m][s] + 1e-12 * np.eye(std[s].size), J)
            Jw = J / std[s][:, None]                                   # whitened sensitivity
            for b in blocks[s]:
                if len(b) > 1:
                    common += len(b) * Jw[b].mean(0) ** 2
                total += (Jw[b] ** 2).sum(0)
    n = len(pts)
    var = {key: np.diag(np.linalg.inv(Fm / n + P)) for key, Fm in F.items()}
    var_exp_only = {m: np.diag(np.linalg.inv(Fm / n + P)) for m, Fm in F_exp_only.items()}
    c_common = common / total

    # observed MCMC widths
    obs_std = {"baseline": base.std(0)}
    for m, p in CHAINS.items():
        obs_std[m] = load_samples(p).std(0)
    prior_std = (hi - lo) / np.sqrt(12)

    rows = []
    for j in range(20):
        row = dict(parameter=names[j], common_mode_share=c_common[j],
                   baseline_std_over_prior=obs_std["baseline"][j] / prior_std[j])
        for m in ("cs", "rbf"):
            row[f"{m}_width_ratio_mcmc"] = obs_std[m][j] / obs_std["baseline"][j]
            row[f"{m}_width_ratio_linear"] = np.sqrt(var[m, 1.0][j] / var["baseline", 1.0][j])
            row[f"{m}_width_ratio_linear_no_emulator"] = np.sqrt(var_exp_only[m][j] / var_exp_only["baseline"][j])
            for k in KS:
                row[f"{m}_width_ratio_k{k:g}"] = np.sqrt(var[m, k][j] / var["baseline", k][j])
        rows.append(row)
    with open(OUT / "correlation_width_test.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    summary = {}
    for m in ("cs", "rbf"):
        mc = np.array([r[f"{m}_width_ratio_mcmc"] for r in rows]); li = np.array([r[f"{m}_width_ratio_linear"] for r in rows])
        summary[m] = dict(
            spearman_linear_vs_mcmc=spearman(li, mc),
            sign_agreement=float(np.mean(np.sign(li - 1) == np.sign(mc - 1))),
            spearman_common_mode_vs_log_ratio_mcmc=spearman(c_common, np.log(mc)),
            spearman_common_mode_vs_log_ratio_linear=spearman(c_common, np.log(li)),
            n_widen_by_k={f"{k:g}": int(sum(r[f"{m}_width_ratio_k{k:g}"] > 1 for r in rows)) for k in KS},
            median_ratio_by_k={f"{k:g}": float(np.median([r[f"{m}_width_ratio_k{k:g}"] for r in rows])) for k in KS})
    scan = {}
    for m in [f"cs_rho{rho:g}" for rho in RHOS] + ["cs", "rbf"]:
        ratios = {k: np.sqrt(var[m, k] / var["baseline", k]) for k in KS}
        scan[m] = {f"{k:g}": dict(median=float(np.median(ratios[k])), n_widen=int((ratios[k] > 1).sum()),
                                  max=float(ratios[k].max()), min=float(ratios[k].min())) for k in KS}
        for j in range(20):
            for k in KS:
                rows[j][f"{m}_k{k:g}"] = float(ratios[k][j])
    summary["rho_k_scan"] = scan
    with open(OUT / "correlation_width_test.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    figs, axs = plt_scan(scan, rows, names, c_common)

    short = [x.replace("bulk_viscosity_3_zeta_over_s_", "zeta_").replace("shear_viscosity_3_eta_over_s_", "eta_")
             .replace("_in_GeV", "") for x in names]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row_i, m in enumerate(("cs", "rbf")):
        mc = np.array([r[f"{m}_width_ratio_mcmc"] for r in rows]); li = np.array([r[f"{m}_width_ratio_linear"] for r in rows])
        ax = axes[row_i, 0]
        ax.scatter(li, mc, c=c_common, cmap="viridis", vmin=0, vmax=1, s=35)
        lim = [min(li.min(), mc.min()) * .9, max(li.max(), mc.max()) * 1.1]
        ax.plot(lim, lim, "k:", lw=.8); ax.axhline(1, color="gray", lw=.5); ax.axvline(1, color="gray", lw=.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        for j in np.argsort(-np.abs(np.log(mc)))[:6]:
            ax.annotate(short[j], (li[j], mc[j]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel="width ratio, linear-Gaussian prediction", ylabel="width ratio, MCMC (std sector / std baseline)",
               title=f"{m.upper()} sector: validation (Spearman {summary[m]['spearman_linear_vs_mcmc']:+.2f})")
        ax = axes[row_i, 1]
        for j in range(20):
            yk = [rows[j][f"{m}_width_ratio_k{k:g}"] for k in KS]
            ax.plot(KS, yk, color=plt.cm.viridis(c_common[j]), lw=1.2)
        ax.axhline(1, color="k", lw=.8); ax.axvline(1, color="gray", ls=":", lw=.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set(xlabel=r"experimental error scale $k$  ($\Sigma_\mathrm{exp}\to k^2\Sigma_\mathrm{exp}$)",
               ylabel="width ratio sector / baseline (linear)",
               title=f"{m.upper()}: does a larger experimental error make correlations widen?")
        ax = axes[row_i, 2]
        ax.scatter(c_common, mc, color="#477fb5" if m == "cs" else "#d17a22", s=35, label="MCMC")
        ax.scatter(c_common, li, facecolor="none", edgecolor="k", s=35, label="linear")
        ax.axhline(1, color="gray", lw=.8); ax.set_yscale("log")
        for j in np.argsort(-np.abs(np.log(mc)))[:6]:
            ax.annotate(short[j], (c_common[j], mc[j]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel=r"common-mode share $c_j$ of the information about $\theta_j$",
               ylabel="width ratio sector / baseline",
               title=f"{m.upper()}: Spearman(c, log ratio MCMC) = {summary[m]['spearman_common_mode_vs_log_ratio_mcmc']:+.2f}")
        ax.legend(fontsize=8)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes[:, :2], fraction=.02, label="common-mode share $c_j$")
    fig.suptitle("Experimental correlations and posterior width: validation, error-scale scan, mechanism")
    fig.savefig(OUT / "correlation_width_test.png", dpi=150); fig.savefig(OUT / "correlation_width_test.pdf")

    print(f"{'parameter':42s} {'c':>5s} {'b/prior':>7s} | {'CS mc':>6s} {'lin':>6s} {'k=4':>6s} | {'RBF mc':>6s} {'lin':>6s} {'k=4':>6s}")
    for r in sorted(rows, key=lambda r: -r["common_mode_share"]):
        print(f"{r['parameter'][:42]:42s} {r['common_mode_share']:5.2f} {r['baseline_std_over_prior']:7.2f} | "
              f"{r['cs_width_ratio_mcmc']:6.2f} {r['cs_width_ratio_linear']:6.2f} {r['cs_width_ratio_k4']:6.2f} | "
              f"{r['rbf_width_ratio_mcmc']:6.2f} {r['rbf_width_ratio_linear']:6.2f} {r['rbf_width_ratio_k4']:6.2f}")
    print(json.dumps({m: {k: (v["median"], v["n_widen"]) for k, v in d.items()} for m, d in scan.items()}, indent=0))
    print(json.dumps({m: {k: v for k, v in summary[m].items() if "spearman" in k or "sign" in k} for m in ("cs", "rbf")}, indent=0))


if __name__ == "__main__":
    main()
