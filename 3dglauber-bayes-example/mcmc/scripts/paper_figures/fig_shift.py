#!/usr/bin/env python3
"""S1-shift: |median_model - median_baseline| / sigma_baseline vs R_j (DESIGN_SYSTEM 2.06).

R_j = sum_i w_ij r_i / sum_i w_ij over all 917 bins, with Fisher weights w_ij and
r_i = sigma_emu^2/sigma_exp^2 from mcmc/02_12_shift_vs_cov_ratio/bin_level.npz
(definitions in mcmc/scripts/shift_vs_cov_ratio.py). Shifts come from the 3dglauber chains only.
Usage: python fig_shift.py --proposal sector --outdir <...>/06_S1-shift
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import common
import style

BIN_LEVEL = common.MCMC / "02_12_shift_vs_cov_ratio/bin_level.npz"
XLIM, YLIM, NLABEL = (1e-1, 1e3), (0, 2), 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", choices=common.PROPOSALS, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    style.apply()
    z = np.load(BIN_LEVEL)
    W, r = z["W"], z["r"]
    R = (W * r).sum(1) / W.sum(1)
    base = common.load("baseline")
    med_b, sd_b = np.median(base, 0), base.std(0, ddof=1)
    models = common.PROPOSALS[a.proposal][1:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    rows = {}
    for ax, m in zip(axes, models):
        spec = common.MODELS[m]
        ax.set_xscale("log")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.axhline(1, color="#888888", ls=(0, (4, 3)), lw=1)
        ax.set_xlabel(r"$R_j=\langle\sigma^2_{\mathrm{emu}}/\sigma^2_{\mathrm{exp}}\rangle_{w_j}$ (917 bins)")
        ax.set_title(spec["label"])
        if not common.available(m):
            ax.text(0.5, 0.5, "chain not available yet", transform=ax.transAxes, ha="center",
                    color="#888888", fontsize=13)
            continue
        s = np.abs(np.median(common.load(m), 0) - med_b) / sd_b
        rho = spearmanr(R, s).statistic
        ax.scatter(R, s, s=36, color=spec["color"], zorder=3)
        for n_, k in enumerate(np.argsort(s)[::-1][:NLABEL]):
            ax.annotate(common.LABELS[k], (R[k], s[k]), xytext=(6, 4 if n_ % 2 == 0 else -12),
                        textcoords="offset points", fontsize=9)
        ax.set_title(f"{spec['label']}  (Spearman = {rho:+.2f})")
        rows[m] = dict(shift=s.tolist(), spearman=float(rho))
    axes[0].set_ylabel(r"|median shift| ($\sigma_{\mathrm{baseline}}$)")
    fig.tight_layout()
    stem = f"S1-shift__{a.proposal}__baseline-{'-'.join(common.SHORT[m] for m in models)}"
    style.save(fig, a.outdir, stem)
    names, _, _ = common.design()
    common.write_manifest(a.outdir, stem, dict(common.provenance(["baseline", *models]),
                                               bin_level=str(BIN_LEVEL.relative_to(common.PROJECT)),
                                               parameters=names, R=R.tolist(), models=rows))


if __name__ == "__main__":
    main()
