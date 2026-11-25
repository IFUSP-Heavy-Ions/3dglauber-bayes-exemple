#!/usr/bin/env python3
"""F14-viscosity: eta/s(T) and zeta/s(T) 90% bands, prior + baseline + models (DESIGN_SYSTEM 2.02).

Eqs. (4)-(6) of Mankolli et al., PRC 114, 014905 (2026).
Usage: python fig_viscosity.py --proposal sector --outdir <img>/2-.../02_F14-viscosity
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import common
import style

T = np.linspace(0.13, 0.30, 200)
I_SHEAR = slice(11, 15)   # T_kink, m_low, m_high, eta/s at kink
I_BULK = slice(15, 19)    # zeta/s max, T_max, width, lambda


def eta_s(p):
    tk, mlo, mhi, ek = (p[:, k][:, None] for k in range(4))
    v = np.where(T[None] < tk, ek + mlo * (tk - T[None]), ek + mhi * (T[None] - tk))
    return np.maximum(v, 0.0)


def zeta_s(p):
    zm, tm, w, lam = (p[:, k][:, None] for k in range(4))
    lam_ = w * (1 + lam * np.sign(T[None] - tm))
    return zm * lam_**2 / (lam_**2 + (T[None] - tm) ** 2)


def band(curves):
    t = (1 - style.CI) / 2
    return np.quantile(curves, [t, 1 - t], axis=0)


def prior_samples(n, rng):
    _, lo, hi = common.design()
    return rng.uniform(lo, hi, size=(n, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", choices=common.PROPOSALS, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    style.apply()
    models = [m for m in common.PROPOSALS[a.proposal] if common.available(m)]
    missing = sorted(set(common.PROPOSALS[a.proposal]) - set(models))
    rng = np.random.default_rng(style.SEED)
    prior = prior_samples(80000, rng)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [(axes[0], eta_s, I_SHEAR, (0, 0.50), r"$\eta/s$"),
              (axes[1], zeta_s, I_BULK, (0, 0.20), r"$\zeta/s$")]
    summary = {}
    for ax, fn, idx, ylim, ylab in panels:
        lo, hi = band(fn(prior[:, idx]))
        ax.fill_between(T, lo, hi, color=style.PRIOR, alpha=0.8, lw=0)
        for m in models:
            s = common.load(m)
            lo, hi = band(fn(s[:, idx]))
            summary.setdefault(m, {})[ylab] = dict(lo=lo.tolist(), hi=hi.tolist())
            spec = common.MODELS[m]
            if m == "baseline":
                ax.fill_between(T, lo, hi, facecolor=(*plt.matplotlib.colors.to_rgb(spec["color"]), 0.45),
                                edgecolor=spec["color"], hatch=style.HATCH, lw=0.6)
            else:
                for y in (lo, hi):
                    ax.plot(T, y, color=spec["color"], ls=spec["ls"], lw=2.5)
        ax.set_xlim(0.13, 0.30)
        ax.set_xticks(np.arange(0.14, 0.301, 0.02))
        ax.set_ylim(*ylim)
        ax.set_xlabel("T [GeV]")
        ax.set_ylabel(ylab)
    handles = [Patch(color=style.PRIOR, alpha=0.8, label="90% Prior")]
    for m in models:
        spec = common.MODELS[m]
        if m == "baseline":
            handles.append(Patch(facecolor=spec["color"], alpha=0.45, hatch=style.HATCH,
                                 edgecolor=spec["color"], label=f"90% C.I. {spec['label']}"))
        else:
            handles.append(Line2D([], [], color=spec["color"], ls=spec["ls"], lw=2.5,
                                  label=f"90% C.I. {spec['label']}"))
    # One legend, in the empty upper-left of eta/s, so no band is hidden (zeta/s upper right is crowded).
    axes[0].legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9)
    fig.tight_layout()
    comp = "-".join(common.SHORT.get(m, m) for m in models)
    stem = f"F14-viscosity__{a.proposal}__{comp}"
    style.save(fig, a.outdir, stem)
    common.write_manifest(a.outdir, stem, dict(common.provenance(models), missing_models=missing,
                                               T_grid=[0.13, 0.30, 200], bands=summary))


if __name__ == "__main__":
    main()
