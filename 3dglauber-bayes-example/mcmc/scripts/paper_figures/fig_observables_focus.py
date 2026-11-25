#!/usr/bin/env python3
"""F05-observables (focus): only the observables touched by the proposal-3 nuisances.

Proposal 3 (covmom) puts Gamma-skewed nuisances lambda_a on Au-Au STAR v2(eta) 20-70% only, so this
figure shows that one observable, in the Fig. 5 layout (main panel + theory/data ratio):
  (a) raw data, 90% bands of every model of the proposal (baseline filled, others as lines);
  (b) nuisance models vs the data corrected by their own posterior shift, y - sum_a lambda_a vhat_a
      (median), bars sqrt(sigma_stat^2 + Var_post[sum_a lambda_a vhat_a]).
Usage: python fig_observables_focus.py --proposal covmom --outdir <...>/03_F05-observables
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common
import emu
import style
from fig_observables import bands, describe, RATIO_LIM

SYSTEM, OBS = "Au-Au-200", "v22_eta_cen_20_70_STAR"
SOURCES = {"nuis_global": common.PROJECT / "study_nongaussian_systematics/data/inferred_sources.npz",
           "nuis_split": common.PROJECT / "study_nongaussian_systematics/data/detector_split/inferred_sources.npz"}


def lambdas(model):
    """Nuisance columns with the same production/thinning policy as common.load."""
    with h5py.File(common.MODELS[model]["path"], "r") as f:
        d = f["chain"]
        blocks = [d[:, s:min(s + 2000, common.N_PROD):common.THIN, 20:] for s in range(0, common.N_PROD, 2000)]
    lam = np.concatenate(blocks, axis=1)
    return lam.reshape(-1, lam.shape[-1])


def corrected(model, y):
    src = np.load(SOURCES[model])
    shift = lambdas(model) @ src["v_hat"].T
    err = np.sqrt(src["sigma_stat"] ** 2 + shift.var(0))
    return y - np.median(shift, 0), err, shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", choices=common.PROPOSALS, default="covmom")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    style.apply()
    models = [m for m in common.PROPOSALS[a.proposal] if common.available(m)]
    rows = [r for r in emu.bins() if r["system"] == SYSTEM and r["observable"] == OBS]
    x = np.array([float(r["center"]) for r in rows])
    y = np.array([float(r["measurement"]) for r in rows])
    dy = np.array([float(r["experimental_std"]) for r in rows])
    b = {m: bands(emu.predictions(m)[SYSTEM][OBS]) for m in models}
    nuis = [m for m in models if m in SOURCES]
    corr = {m: corrected(m, y) for m in nuis}

    lo = min([np.min(y - dy)] + [v[0].min() for v in b.values()] + [np.min(c - e) for c, e, _ in corr.values()])
    hi = max([np.max(y + dy)] + [v[1].max() for v in b.values()] + [np.max(c + e) for c, e, _ in corr.values()])
    pad = 0.05 * (hi - lo)
    fig = plt.figure(figsize=(12, 5.2))
    outer = fig.add_gridspec(1, 2, wspace=0.22, left=0.07, right=0.99, top=0.88, bottom=0.12)
    titles = ["(a) raw data, all models", "(b) skew nuisances vs data corrected by their posterior shift"]
    for k, title in enumerate(titles):
        inner = outer[k].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax, rx = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        show = models if k == 0 else ["baseline", *nuis]
        for m in show:
            spec, (l, h) = common.MODELS[m], b[m]
            if m == "baseline":
                ax.fill_between(x, l, h, color=spec["color"], alpha=0.6, lw=0)
                rx.fill_between(x, l / y, h / y, color=spec["color"], alpha=0.6, lw=0)
            else:
                for v in (l, h):
                    ax.plot(x, v, color=spec["color"], ls=spec["ls"], lw=1.8)
                    rx.plot(x, v / y, color=spec["color"], ls=spec["ls"], lw=1.5)
        if k == 0:
            ax.errorbar(x, y, dy, fmt="o", color="black", ms=4, elinewidth=0.8, capsize=0)
            rx.errorbar(x, np.ones_like(y), dy / y, fmt="o", color="black", ms=3, elinewidth=0.7, capsize=0)
        else:
            ax.errorbar(x, y, dy, fmt="o", mfc="none", mec="#999999", ecolor="#bbbbbb", ms=4, elinewidth=0.6, capsize=0)
            for j, m in enumerate(nuis):
                c, e, _ = corr[m]
                off = (j - (len(nuis) - 1) / 2) * 0.06
                col = common.MODELS[m]["color"]
                ax.errorbar(x + off, c, e, fmt="s", color=col, ms=4, elinewidth=0.9, capsize=0)
                rx.errorbar(x + off, c / y, e / y, fmt="s", color=col, ms=3, elinewidth=0.8, capsize=0)
        ax.set_ylim(lo - pad, hi + pad)
        rx.set_ylim(*RATIO_LIM)
        rx.axhline(1, color="#777777", lw=0.6)
        ax.tick_params(labelbottom=False)
        rx.set_xlabel(r"$\eta$")
        rx.set_ylabel("theory/data", fontsize=10)
        ax.set_ylabel(r"$v_2\{2\}$")
        ax.set_title(title, fontsize=12)
        sysname, name, exp = describe(SYSTEM, OBS)
        ax.text(0.03, 0.97, sysname, transform=ax.transAxes, va="top", fontsize=10, weight="bold")
        ax.text(0.03, 0.88, name, transform=ax.transAxes, va="top", fontsize=10, style="italic")
        ax.text(0.03, 0.79, exp, transform=ax.transAxes, va="top", fontsize=10, weight="bold", style="italic")
        handles = []
        for m in show:
            spec = common.MODELS[m]
            handles.append(Patch(color=spec["color"], alpha=0.6, label=f"90% C.I. {spec['label']}") if m == "baseline"
                           else Line2D([], [], color=spec["color"], ls=spec["ls"], lw=1.8, label=f"90% C.I. {spec['label']}"))
        if k == 0:
            handles.append(Line2D([], [], color="black", marker="o", ls="", label="data"))
        else:
            handles.append(Line2D([], [], color="#999999", marker="o", mfc="none", ls="", label="data (raw)"))
            handles += [Line2D([], [], color=common.MODELS[m]["color"], marker="s", ls="",
                               label=f"data corrected, {common.MODELS[m]['label']}") for m in nuis]
        ax.legend(handles=handles, loc="lower center", fontsize=8.5 if k == 0 else 7.2, frameon=True, framealpha=0.9)
    comp = "-".join(common.SHORT.get(m, m) for m in models)
    stem = f"F05-observables__{a.proposal}__{comp}__star-v2eta-20-70"
    style.save(fig, a.outdir, stem)
    common.write_manifest(a.outdir, stem, dict(
        common.provenance(models), observable=f"{SYSTEM}/{OBS}", n_prediction_samples=emu.N_PRED,
        sources={m: str(p.relative_to(common.PROJECT)) for m, p in SOURCES.items()},
        corrected_data={m: dict(y=c.tolist(), err=e.tolist(), shift_median=np.median(s, 0).tolist())
                        for m, (c, e, s) in corr.items()},
        bands="emulator mean only (no emulator or experimental noise draws), as in the paper"))


if __name__ == "__main__":
    main()
