#!/usr/bin/env python3
"""S2-expcorr: experimental covariance (top) and correlation (bottom) over all 917 bins, one column per model
(DESIGN_SYSTEM 2.07).

Panels: baseline (identity) | CS | RBF of the proposal. Bin order = likelihood order
(mcmc/02_08_sector_input_correlations/bin_index.csv; Au-Au 791 bins then d-Au 126).
  - sector CS / RBF: reconstructed matrices in mcmc/02_08_sector_input_correlations/*_input_matrices.npz
  - CS global (scope=system): rho = 1 inside each system, 0 across systems
  - RBF global: experimental_global_covariance with global_rbf_acceptance.json and the lengths of the run
Covariance in likelihood coordinates: sigma_i = likelihood_std of bin_index.csv (log for multiplicities), cov =
sigma_i sigma_j R_ij, log colour scale with the same range in every panel. Correlation: cividis, fixed [0, 1].
Own colorbar per panel (each panel can be cropped). Thin white lines between observables, thick black line
between systems.
Usage: python fig_expcorr.py --proposal sector --outdir <...>/07_S2-expcorr
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import common
import emu
import style


def _load_global_cov():
    """Load by path: over SSHFS the import system can miss newly created files (stale dir cache)."""
    import importlib.util
    sys.path.insert(0, str(common.PROJECT))
    spec = importlib.util.spec_from_file_location(
        "experimental_global_covariance", common.PROJECT / "experimental_global_covariance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_G = _load_global_cov()
AXES, bin_coordinates, load_acceptance = _G.AXES, _G.bin_coordinates, _G.load_acceptance

SECTOR_NPZ = {"cs_sector": common.MCMC / "02_08_sector_input_correlations/cs_sector_input_matrices.npz",
              "rbf_sector": common.MCMC / "02_08_sector_input_correlations/rbf_sector_input_matrices.npz"}
RBF_GLOBAL_LENGTHS = {"pt": 0.5, "eta": 1.0, "centrality": 10.0}  # as submitted (condor_rbf_global.sub)
LEARNED_PHOBOS = {"u": 0.0148, "slope": 0.025}  # as submitted (condor_learned_phobos.sub)


def groups(rows):
    out = []
    for i, r in enumerate(rows):
        key = (r["system"], r["observable"])
        if not out or out[-1][0] != key:
            out.append([key, i, i + 1])
        else:
            out[-1][2] = i + 1
    return out


def correlation(model, rows):
    n = len(rows)
    systems = np.array([r["system"] for r in rows])
    if model == "baseline":
        return np.eye(n)
    if model in SECTOR_NPZ:
        return np.load(SECTOR_NPZ[model])["correlation"]
    if model == "cs_global":
        return (systems[:, None] == systems[None, :]).astype(float)
    if model == "rbf_global":
        acc = load_acceptance()
        coords = np.empty((n, 3))
        for (system, obs), a, b in groups(rows):
            edges = np.array([[float(r["bin_low"]), float(r["bin_high"])] for r in rows[a:b]])
            coords[a:b] = bin_coordinates(system, obs, edges, acc)
        scaled = coords / np.array([RBF_GLOBAL_LENGTHS[k] for k in AXES])
        r = np.exp(-((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(-1))
        return r * (systems[:, None] == systems[None, :])
    if model == "learned_phobos":
        # Same construction as the 02_14 chain (experimental_learned_covariance.py, u and slope of the job),
        # evaluated with the likelihood sigmas of bin_index.csv; Au-Au and d-Au uncorrelated.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "experimental_learned_covariance", common.PROJECT / "experimental_learned_covariance.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = np.zeros((n, n))
        for system in dict.fromkeys(systems):
            idx = np.flatnonzero(systems == system)
            start = idx[0]
            slices = [(obs, slice(a - start, b - start)) for (s_, obs), a, b in groups(rows) if s_ == system]
            std = np.array([float(rows[i]["likelihood_std"]) for i in idx])
            cov = mod.learned_phobos_covariance(system, slices, std, **LEARNED_PHOBOS)
            out[np.ix_(idx, idx)] = cov / np.outer(std, std)
        return out
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", choices=common.PROPOSALS, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    style.apply()
    rows = emu.bins()
    grp = groups(rows)
    n_au = sum(1 for r in rows if r["system"] == "Au-Au-200")
    models = common.PROPOSALS[a.proposal]
    from matplotlib.colors import LogNorm
    sig = np.array([float(r["likelihood_std"]) for r in rows])
    corr = {m: correlation(m, rows) for m in models}
    cov = {m: np.outer(sig, sig) * corr[m] for m in models}
    vmax = max(c.max() for c in cov.values())
    vmin = 10 ** np.floor(np.log10(np.min(sig ** 2)))
    fig, axes = plt.subplots(2, 3, figsize=(27, 17.5))
    ticks = [n_au / 2, n_au + (len(rows) - n_au) / 2]
    stats = {}
    for col, m in enumerate(models):
        for row, (mat, norm, label, what) in enumerate((
                (cov[m], LogNorm(vmin=vmin, vmax=vmax), r"$\mathrm{Cov}(y_i,y_j)$ (likelihood units)", "covariance"),
                (np.clip(corr[m], 0, 1), plt.Normalize(0, 1), r"$\mathrm{Cov}(y_i,y_j)/\sigma_i\sigma_j$", "correlation"))):
            ax = axes[row, col]
            shown = np.where(mat > 0, mat, np.nan) if row == 0 else mat
            im = ax.imshow(shown, cmap="cividis", norm=norm, interpolation="nearest", origin="lower")  # bin 0 meets bin 0 (bottom-left)
            # Observable separators: barely visible; system separator: discreet (DESIGN_SYSTEM 2.07).
            for _, s0, _ in grp[1:]:
                ax.axhline(s0 - 0.5, color="#c8c8c8", lw=0.15, alpha=0.25)
                ax.axvline(s0 - 0.5, color="#c8c8c8", lw=0.15, alpha=0.25)
            ax.axhline(n_au - 0.5, color="#9a9a9a", lw=0.6, alpha=0.6)
            ax.axvline(n_au - 0.5, color="#9a9a9a", lw=0.6, alpha=0.6)
            # Bin counter 0..917 on both axes; system names on the opposite axes.
            counter = [c for c in range(0, len(rows), 100) if len(rows) - c >= 50] + [len(rows)]
            ax.set_xticks([c - 0.5 for c in counter], [str(c) for c in counter])
            ax.set_yticks([c - 0.5 for c in counter], [str(c) for c in counter])
            ax.set_xlabel("bin index")
            ax.set_ylabel("bin index")
            top = ax.secondary_xaxis("top")
            top.set_xticks(ticks, ["Au-Au 200", "d-Au 200"])
            top.tick_params(length=0)
            right = ax.secondary_yaxis("right")
            right.set_yticks(ticks, ["Au-Au 200", "d-Au 200"], rotation=90, va="center")
            right.tick_params(length=0)
            ax.set_title(f"exp. {what}: {common.MODELS[m]['label']}", pad=28)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.07)
            cb.set_label(label)
        off = corr[m][~np.eye(len(rows), dtype=bool)]
        stats[m] = dict(mean_offdiag_correlation=float(off.mean()), frac_above_0p5=float((off > 0.5).mean()))
    fig.tight_layout()
    stem = f"S2-expcov__{a.proposal}__{'-'.join(common.SHORT.get(m, m) for m in models)}"
    style.save(fig, a.outdir, stem)
    common.write_manifest(a.outdir, stem, dict(
        models=models, n_bins=len(rows), bin_index=str(emu.BIN_INDEX.relative_to(common.PROJECT)),
        sector_sources={k: str(v.relative_to(common.PROJECT)) for k, v in SECTOR_NPZ.items()},
        rbf_global_lengths=RBF_GLOBAL_LENGTHS, learned_phobos=LEARNED_PHOBOS, covariance_log_range=[float(vmin), float(vmax)],
        correlation_range=[0, 1], statistics=stats,
        observable_order=[f"{s}/{o}" for (s, o), _, _ in grp]))


if __name__ == "__main__":
    main()
