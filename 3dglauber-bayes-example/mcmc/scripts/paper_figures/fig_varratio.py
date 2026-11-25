#!/usr/bin/env python3
"""F25-varratio: diagonal Sigma_emu/Sigma_exp for the three observables of Fig. 25 (DESIGN_SYSTEM 2.04).

r_i = sigma_emu,i^2 / sigma_exp,i^2 from mcmc/02_12_shift_vs_cov_ratio/bin_level.npz (averaged over
60 baseline-posterior samples, likelihood coordinates). The correlation models keep the diagonal
of Sigma_exp, so this figure is identical for every proposal.
Usage: python fig_varratio.py --outdir <...>/04_F25-varratio
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import common
import style

BIN_LEVEL = common.MCMC / "02_12_shift_vs_cov_ratio/bin_level.npz"
BIN_INDEX = common.MCMC / "02_08_sector_input_correlations/bin_index.csv"
OBSERVABLES = [
    ("d-Au-200", "dNdeta_eta_cen_00_05_PHEN", r"d-Au $dN/d\eta$ 0-5%", r"$\eta$"),
    ("Au-Au-200", "v22_eta_cen_03_15_PHOB", r"Au-Au $v_2(\eta)$ 3-15%", r"$\eta$"),
    ("Au-Au-200", "v22_pt_cen_20_30_PHEN", r"Au-Au $v_2(p_T)$ 20-30%", r"$p_T$ (GeV/c)"),
]


def bins():
    with BIN_INDEX.open() as f:
        return list(csv.DictReader(f))


def select(rows, system, obs):
    idx = [int(r["global_index"]) for r in rows if r["system"] == system and r["observable"] == obs]
    centers = [float(r["center"]) for r in rows if r["system"] == system and r["observable"] == obs]
    return np.array(idx), np.array(centers)


def ticks(ax, centers, n=6):
    k = np.unique(np.linspace(0, len(centers) - 1, min(n, len(centers))).round().astype(int))
    ax.set_xticks(k)
    ax.set_xticklabels([f"{centers[i]:.1f}" for i in k])
    ax.set_yticks(k)
    ax.set_yticklabels([f"{centers[i]:.1f}" for i in k])


def all_observables(a, r, rows):
    """All 40 calibrated observables, F05 grid/order. Each panel has its own linear colour scale and
    colorbar, as in Fig. 25, so any panel can be cropped and used on its own."""
    from fig_observables import NROWS, NCOLS, XLAB, describe, order
    panels = order(rows)
    fig = plt.figure(figsize=(22, 24))
    grid = fig.add_gridspec(NROWS, NCOLS, hspace=0.55, wspace=0.55, left=0.04, right=0.98, top=0.97, bottom=0.04)
    out = {}
    for k, (system, obs) in enumerate(panels):
        ax = fig.add_subplot(grid[k // NCOLS, k % NCOLS])
        idx, centers = select(rows, system, obs)
        m = np.full((len(idx), len(idx)), np.nan)
        np.fill_diagonal(m, r[idx])
        im = ax.imshow(m, cmap="cividis", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)
        ticks(ax, centers, n=4)
        ax.tick_params(labelsize=8)
        coord = next(x["coordinate"] for x in rows if x["system"] == system and x["observable"] == obs)
        ax.set_xlabel(XLAB[coord], fontsize=10)
        sysname, name, exp = describe(system, obs)
        ax.set_title(r"$\Sigma_{\mathrm{emu}}/\Sigma_{\mathrm{exp}}$ " + f"{sysname.replace(' 200 GeV', '')}\n{name} {exp}",
                     fontsize=9)
        out[f"{system}/{obs}"] = dict(centers=centers.tolist(), ratio=r[idx].tolist())
    stem = f"F25-varratio__{a.proposal}__baseline__all"
    style.save(fig, a.outdir, stem)
    common.write_manifest(a.outdir, stem, dict(source=str(BIN_LEVEL.relative_to(common.PROJECT)),
                                               colour_scale="per panel, linear, own colorbar (as Fig. 25)",
                                               panel_order=[f"{s}/{o}" for s, o in panels], ratios=out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--proposal", default="all", help="tag used in the file name only")
    a = ap.parse_args()
    style.apply()
    r = np.load(BIN_LEVEL)["r"]
    rows = bins()
    all_observables(a, r, rows)
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 15))
    out = {}
    for ax, (system, obs, title, xlab) in zip(axes, OBSERVABLES):
        idx, centers = select(rows, system, obs)
        m = np.full((len(idx), len(idx)), np.nan)
        np.fill_diagonal(m, r[idx])
        im = ax.imshow(m, cmap="cividis", origin="upper", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(r"$\Sigma_{\mathrm{emu}}\,/\,\Sigma_{\mathrm{exp}}$ (" + title + ")")
        ticks(ax, centers)
        ax.set_xlabel(xlab)
        ax.set_ylabel(xlab)
        out[f"{system}/{obs}"] = dict(centers=centers.tolist(), ratio=r[idx].tolist())
    fig.tight_layout()
    stem = f"F25-varratio__{a.proposal}__baseline"
    style.save(fig, a.outdir, stem)
    common.write_manifest(a.outdir, stem, dict(source=str(BIN_LEVEL.relative_to(common.PROJECT)),
                                               bins=str(BIN_INDEX.relative_to(common.PROJECT)), ratios=out))


if __name__ == "__main__":
    main()
