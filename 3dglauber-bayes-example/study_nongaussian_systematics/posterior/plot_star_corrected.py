#!/usr/bin/env python3
"""STAR v2(eta) with the data corrected by the inferred systematic shift, with honest error bars.

Companion to compare_posteriors.py (which is kept unchanged). There, the green points are
y_STAR - sum_a lambda_a vhat_a with bars = 68% posterior spread of the correction only.
Here the corrected points carry the uncertainty that remains in the nuisance model:

    thick bar:  sigma_stat  (what stays in the likelihood covariance after removing
                             the correlated systematic part)
    thin bar:   sqrt(sigma_stat^2 + Var_post[sum_a lambda_a vhat_a])

The raw STAR points keep the reported total error. Only saved outputs are read
(no emulator): <dir>/nuisance_posterior_inputs.npz, <dir>/nuisance_predictions_Au-Au-200.npz,
the baseline/CS saved predictions and the inferred_sources.npz used by the chain.

Usage (from 3dglauber-bayes-example):
    python study_nongaussian_systematics/posterior/plot_star_corrected.py               # 02_10 (3 models)
    python study_nongaussian_systematics/posterior/plot_star_corrected.py \
        --dir mcmc/02_11_nongaussian_nuisance_detector_split \
        --sources study_nongaussian_systematics/data/detector_split/inferred_sources.npz \
        --label "Nuisance, TPC/FTPC normalizations" --previous mcmc/02_10_nongaussian_nuisance
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SYSTEM, OBS = "Au-Au-200", "v22_eta_cen_20_70_STAR"
BASE_PRED = ROOT / "mcmc/02_06_baseline_vs_cs_sector/baseline_predictions"
CS_PRED = ROOT / "mcmc/02_04_cs_sector/plots/observables"
DEFAULT_DIR = ROOT / "mcmc/02_10_nongaussian_nuisance"
DEFAULT_SOURCES = ROOT / "study_nongaussian_systematics/data/inferred_sources.npz"
REGIONS = [("backward_FTPC", slice(0, 6)), ("TPC", slice(6, 15)), ("forward_FTPC", slice(15, 21))]
Q = [.16, .5, .84]


def load_data():
    rows = [l.split() for l in (ROOT / "HIC_experimental_data/Au-Au-200/STAR" / f"{OBS}.dat").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    return np.array(rows, float)[:, 0]


def corrected(data, folder, sources):
    """Corrected points and the two error bars from a nuisance comparison folder."""
    src = np.load(sources)
    lam = np.load(folder / "nuisance_posterior_inputs.npz")["lam"]
    if lam.shape[1] != src["v_hat"].shape[1]:
        raise ValueError(f"{folder}: {lam.shape[1]} lambdas but {src['v_hat'].shape[1]} sources in {sources}")
    shift = lam @ src["v_hat"].T                                    # (samples, 21)
    point = data - np.median(shift, axis=0)
    sig_stat = np.asarray(src["sigma_stat"], float)
    sig_full = np.sqrt(sig_stat**2 + shift.var(axis=0))
    return point, sig_stat, sig_full, np.asarray(src["sigma_total"], float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="nuisance comparison folder")
    ap.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--label", default="Non-Gaussian nuisance")
    ap.add_argument("--previous", type=Path, help="earlier nuisance folder (global normalization) to overlay")
    ap.add_argument("--previous-sources", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--previous-label", default="Nuisance, global normalization")
    args = ap.parse_args()
    folder = args.dir if args.dir.is_absolute() else ROOT / args.dir
    sources = args.sources if args.sources.is_absolute() else ROOT / args.sources

    data = load_data()
    eta = np.load(sources)["eta"]
    preds = {"baseline": np.load(BASE_PRED / f"predictions_{SYSTEM}.npz")[OBS],
             "cs": np.load(CS_PRED / f"predictions_{SYSTEM}.npz")[OBS],
             "nuis": np.load(folder / f"nuisance_predictions_{SYSTEM}.npz")[OBS]}
    models = [("baseline", "Baseline (diagonal)", "#df9393", "--"),
              ("cs", "CS sector (rho=1)", "#477fb5", "-"),
              ("nuis", args.label, "#2a9d5c", "-")]
    corr = [("nuis", args.label, "#2a9d5c", "D", +0.07, corrected(data, folder, sources))]
    if args.previous:
        prev = args.previous if args.previous.is_absolute() else ROOT / args.previous
        prev_src = args.previous_sources if args.previous_sources.is_absolute() else ROOT / args.previous_sources
        preds["nuis_prev"] = np.load(prev / f"nuisance_predictions_{SYSTEM}.npz")[OBS]
        models.insert(2, ("nuis_prev", args.previous_label, "#8c6bb1", ":"))
        corr.insert(0, ("nuis_prev", args.previous_label, "#8c6bb1", "s", -0.07, corrected(data, prev, prev_src)))
    tag = f"{len(models)}way"
    sig_tot = corr[-1][5][3]

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
    for key, label, color, ls in models:
        q = np.quantile(preds[key], Q, axis=0)
        for _, slc in REGIONS:
            ax.fill_between(eta[slc], q[0, slc], q[2, slc], color=color, alpha=.22)
            ax.plot(eta[slc], q[1, slc], color=color, ls=ls, lw=1.6, label=label if slc.start == 0 else None)
            rax.fill_between(eta[slc], q[0, slc] / data[slc], q[2, slc] / data[slc], color=color, alpha=.2)
            rax.plot(eta[slc], q[1, slc] / data[slc], color=color, ls=ls, lw=1.4)
    ax.errorbar(eta, data, yerr=sig_tot, fmt="o", color="k", ms=4, lw=1, label=r"STAR ($\sigma_\mathrm{total}$)")
    rax.errorbar(eta, np.ones_like(data), yerr=sig_tot / data, fmt="o", color="k", ms=3, lw=.8)
    for key, label, color, marker, dx, (point, s_stat, s_full, _) in corr:
        x = eta + dx
        ax.errorbar(x, point, yerr=s_full, fmt="none", ecolor=color, lw=.8, capsize=2)
        ax.errorbar(x, point, yerr=s_stat, fmt=marker, color=color, mfc="white", ms=4, lw=2.4,
                    label=rf"STAR $-\sum_a\lambda_a\hat v_a$ [{label}]")
        rax.errorbar(x, point / data, yerr=s_full / data, fmt="none", ecolor=color, lw=.8, capsize=2)
        rax.errorbar(x, point / data, yerr=s_stat / data, fmt=marker, color=color, mfc="white", ms=3, lw=2)
    ax.plot([], [], color="gray", lw=2.4, label=r"thick: $\sigma_\mathrm{stat}$ (likelihood covariance)")
    ax.plot([], [], color="gray", lw=.8, label=r"thin: $\sqrt{\sigma_\mathrm{stat}^2+\mathrm{Var}_\mathrm{post}(\sum_a\lambda_a\hat v_a)}$")
    ax.set_ylabel(r"$v_2\{2\}$"); ax.legend(fontsize=7, frameon=False, ncol=1, loc="lower center")
    ax.set_title("STAR Au+Au 200 GeV, 20-70%: data corrected by the inferred systematic shift")
    rax.axhline(1, color="gray", lw=.8); rax.set(xlabel=r"$\eta$", ylabel="model / data")
    fig.tight_layout()
    stem = folder / f"star_v2_eta_corrected_{tag}"
    fig.savefig(f"{stem}.png", dpi=160); fig.savefig(f"{stem}.pdf"); plt.close(fig)

    # how far the corrected points are from each model median, in units of the corrected error
    summary = {}
    for key, label, *_ , (point, s_stat, s_full, _) in corr:
        for mkey, *_ in models:
            med = np.median(preds[mkey], axis=0)
            for name, slc in REGIONS:
                summary[f"corr[{key}]_vs_{mkey}_{name}_pull_rms"] = float(
                    np.sqrt(np.mean(((med[slc] - point[slc]) / s_full[slc]) ** 2)))
        for name, slc in REGIONS:
            summary[f"corr[{key}]_{name}_sigma_stat_over_total"] = float(np.mean(s_stat[slc] / sig_tot[slc]))
    (folder / f"star_v2_eta_corrected_{tag}.json").write_text(json.dumps(summary, indent=1))
    print("wrote", f"{stem}.png"); print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
