#!/usr/bin/env python3
"""F26-corrmatrix: emulator vs experimental correlation for the Fig. 26 observables (DESIGN_SYSTEM 2.05).

Columns: emulator | exp. baseline (identity) | exp. CS | exp. RBF. cividis, fixed [0, 1].
Every panel carries its own colorbar and a two-line title (observable / matrix), so any panel can be
cropped and read on its own. Layout is constrained so x labels never overlap the titles below.
Experimental correlation:
  - sector: reconstructed matrices in mcmc/02_08_sector_input_correlations/*_sector_input_matrices.npz
  - global CS (scope=system): every pair of bins in the same system has rho = 1
  - global RBF: experimental_global_covariance.py + global_rbf_acceptance.json (lengths of the run)
White lines separate rapidity regions (sector column of bin_index.csv), as in Fig. 26.
Usage: python fig_corrmatrix.py --proposal sector --outdir <...>/05_F26-corrmatrix
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import common
import emu
import style
from fig_varratio import OBSERVABLES, ticks

SECTOR_NPZ = {"cs_sector": common.MCMC / "02_08_sector_input_correlations/cs_sector_input_matrices.npz",
              "rbf_sector": common.MCMC / "02_08_sector_input_correlations/rbf_sector_input_matrices.npz"}


def exp_correlation(model, idx, rows):
    if model in SECTOR_NPZ:
        return np.load(SECTOR_NPZ[model])["correlation"][np.ix_(idx, idx)]
    if model == "cs_global":
        return np.ones((len(idx), len(idx)))
    if model == "rbf_global":
        from fig_expcorr import correlation
        return correlation("rbf_global", rows)[np.ix_(idx, idx)]
    return None


def region_breaks(rows, idx):
    labels = [rows[i]["sector"] for i in idx]
    return [k for k in range(1, len(labels)) if labels[k] != labels[k - 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", choices=common.PROPOSALS, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    style.apply()
    rows = emu.bins()
    models = common.PROPOSALS[a.proposal][1:]
    cols = ["emulator", "baseline", *models]
    titles = {"emulator": "emulator", "baseline": "exp. baseline (diag)",
              **{m: f"exp. {common.MODELS[m]['label']}" for m in models}}
    fig, axes = plt.subplots(3, 4, figsize=(24, 17), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.25, w_pad=0.15, hspace=0.06, wspace=0.04)
    out = {}
    for r, (system, obs, label, xlab) in enumerate(OBSERVABLES):
        idx = np.array([int(x["global_index"]) for x in rows if x["system"] == system and x["observable"] == obs])
        centers = np.array([float(x["center"]) for x in rows if x["system"] == system and x["observable"] == obs])
        breaks = region_breaks(rows, idx)
        mats = {"emulator": emu.emulator_correlation(system, obs), "baseline": np.eye(len(idx))}
        for m in models:
            mats[m] = exp_correlation(m, idx, rows)
        for c, key in enumerate(cols):
            ax = axes[r, c]
            mat = mats[key]
            ax.set_title(r"$\mathrm{Cov}(y_i,y_j)/\sigma_i\sigma_j$ (" + label + ")\n" + titles[key], fontsize=12)
            if mat is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center",
                        transform=ax.transAxes, color="#888888", fontsize=13)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            im = ax.imshow(np.clip(mat, 0, 1), cmap="cividis", vmin=0, vmax=1, interpolation="nearest")
            for b in breaks:
                ax.axhline(b - 0.5, color="white", lw=4)
                ax.axvline(b - 0.5, color="white", lw=4)
            ticks(ax, centers)
            ax.set_xlabel(xlab)
            ax.set_ylabel(xlab)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=9)
            out.setdefault(f"{system}/{obs}", {})[key] = np.round(mat, 4).tolist()
    stem = f"F26-corrmatrix__{a.proposal}__baseline-cs-rbf"
    style.save(fig, a.outdir, stem)
    common.write_manifest(a.outdir, stem, dict(
        common.provenance(["baseline"]), emulator_points=60, clipped_to=[0, 1],
        sector_sources={k: str(v.relative_to(common.PROJECT)) for k, v in SECTOR_NPZ.items()},
        correlations=out))


if __name__ == "__main__":
    main()
