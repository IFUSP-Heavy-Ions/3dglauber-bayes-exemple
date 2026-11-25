#!/usr/bin/env python3
"""F05-observables: 90% posterior bands of all 40 calibrated observables (917 bins), baseline vs one
model, Fig. 5 layout (main panel + theory/data ratio) on one 7x6 page (DESIGN_SYSTEM 2.03).

Panel order follows Fig. 5 (Au-Au dN/deta, d-Au dN/deta, v2(eta), v2(pT), integrated vn, <pT>).
Y limits: computed once over data +- error and every available model, saved in the manifest, so
pages of different proposals are directly comparable.
Usage: python fig_observables.py --proposal sector --model cs_sector --outdir <...>/03_F05-observables
"""
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common
import emu
import style

NROWS, NCOLS = 7, 6
RATIO_LIM = (0.5, 1.5)
EXP = {"PHOB": "PHOBOS", "PHEN": "PHENIX", "STAR": "STAR", "BRAH": "BRAHMS"}
XLAB = {"eta": r"$\eta$", "pt": r"$p_T$ [GeV/c]", "centrality": "centrality (%)"}


def order(rows):
    seen = []
    for r in rows:
        key = (r["system"], r["observable"])
        if key not in seen:
            seen.append(key)

    def rank(k):
        system, obs = k
        groups = [("Au-Au-200", "dNdeta"), ("d-Au-200", "dNdeta"), ("Au-Au-200", "v22_eta"),
                  ("d-Au-200", "v22_eta"), ("Au-Au-200", "v22_pt"), ("d-Au-200", "v22_pt"),
                  ("Au-Au-200", "_int_"), ("Au-Au-200", "meanpT")]
        g = next(i for i, (s, p) in enumerate(groups) if s == system and p in obs)
        return (g, "STAR" in obs if "v22_pt" in obs else 0, seen.index(k))
    return sorted(seen, key=rank)


def describe(system, obs):
    sysname = system.replace("-200", "") + " 200 GeV"
    exp = EXP[obs.rsplit("_", 1)[1]]
    cen = re.search(r"cen_(\d+)_(\d+)", obs)
    cent = f" {int(cen.group(1))}-{int(cen.group(2))}%" if cen else ""
    if obs.startswith("dNdeta"):
        name = r"$dN/d\eta(\eta)$"
    elif obs.startswith("v22_eta"):
        name = r"$v_2(\eta)$"
    elif obs.startswith("v22_pt"):
        name = r"$v_2(p_T)$"
    elif obs.startswith("v22_int"):
        name = r"$v_2$(cent)"
    elif obs.startswith("v32_int"):
        name = r"$v_3$(cent)"
    else:
        sp = {"pi": r"$\pi^{\pm}$", "k": r"$K^{\pm}$", "p": r"$p/\bar p$"}[obs.split("_")[1]]
        name = sp + r" mean $p_T$(cent)"
    return sysname, name + cent, exp


def bands(pred):
    t = (1 - style.CI) / 2
    return np.quantile(pred, [t, 1 - t], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", choices=common.PROPOSALS, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    style.apply()
    rows = emu.bins()
    panels = order(rows)
    available = [m for m in common.MODELS if common.available(m) and not common.MODELS[m].get("focus")]
    preds = {m: emu.predictions(m) for m in available}
    spec = common.MODELS[a.model]

    fig = plt.figure(figsize=(18, 24))
    outer = fig.add_gridspec(NROWS, NCOLS, hspace=0.38, wspace=0.32, left=0.05, right=0.99, top=0.97, bottom=0.04)
    ylims = {}
    for k, (system, obs) in enumerate(panels):
        sel = [r for r in rows if r["system"] == system and r["observable"] == obs]
        x = np.array([float(r["center"]) for r in sel])
        y = np.array([float(r["measurement"]) for r in sel])
        dy = np.array([float(r["experimental_std"]) for r in sel])
        lo_all = [y - dy]
        hi_all = [y + dy]
        for m in available:
            b = bands(preds[m][system][obs])
            lo_all.append(b[0]); hi_all.append(b[1])
        ymin, ymax = min(map(np.min, lo_all)), max(map(np.max, hi_all))
        pad = 0.05 * (ymax - ymin)
        ylims[f"{system}/{obs}"] = (ymin - pad, ymax + pad)

        inner = outer[k // NCOLS, k % NCOLS].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax, rx = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        bb = bands(preds["baseline"][system][obs])
        ax.fill_between(x, *bb, color=style.BASELINE, alpha=0.6, lw=0, step=None)
        rx.fill_between(x, bb[0] / y, bb[1] / y, color=style.BASELINE, alpha=0.6, lw=0)
        if a.model in preds:
            bm = bands(preds[a.model][system][obs])
            for v in bm:
                ax.plot(x, v, color=spec["color"], ls=spec["ls"], lw=1.6)
                rx.plot(x, v / y, color=spec["color"], ls=spec["ls"], lw=1.4)
        ax.errorbar(x, y, dy, fmt="o", color="black", ms=3.5, elinewidth=0.8, capsize=0)
        rx.errorbar(x, np.ones_like(y), dy / y, fmt="o", color="black", ms=2.5, elinewidth=0.7, capsize=0)
        ax.set_ylim(*ylims[f"{system}/{obs}"])
        rx.set_ylim(*RATIO_LIM)
        rx.axhline(1, color="#777777", lw=0.6)
        ax.tick_params(labelbottom=False, labelsize=9)
        rx.tick_params(labelsize=9)
        rx.set_xlabel(XLAB[sel[0]["coordinate"]], fontsize=11)
        if k % NCOLS == 0:
            rx.set_ylabel("theory/data", fontsize=9)
        sysname, name, exp = describe(system, obs)
        ax.text(0.03, 0.97, sysname, transform=ax.transAxes, va="top", fontsize=10, weight="bold")
        ax.text(0.03, 0.84, name, transform=ax.transAxes, va="top", fontsize=10, style="italic")
        ax.text(0.03, 0.71, exp, transform=ax.transAxes, va="top", fontsize=10, weight="bold", style="italic")
        ax.text(0.5, 0.04, f"{len(x)} bins", transform=ax.transAxes, ha="center", fontsize=10, weight="bold")
    legend = fig.add_subplot(outer[NROWS - 1, NCOLS - 2:])
    legend.axis("off")
    legend.legend(handles=[Patch(color=style.BASELINE, alpha=0.6, label="90% C.I. baseline (diag)"),
                           Line2D([], [], color=spec["color"], ls=spec["ls"], lw=1.6,
                                  label=f"90% C.I. {spec['label']}"),
                           Line2D([], [], color="black", marker="o", ls="", label="data")],
                  loc="center", fontsize=14, frameon=False)
    stem = f"F05-observables__{a.proposal}__baseline-vs-{common.SHORT[a.model]}__p01"
    style.save(fig, a.outdir, stem)
    common.write_manifest(a.outdir, stem, dict(
        common.provenance(["baseline", a.model]), n_prediction_samples=emu.N_PRED,
        model_available=a.model in preds, ylim_models=available, ylims=ylims,
        bands="emulator mean only (no emulator or experimental noise draws), as in the paper",
        panel_order=[f"{s}/{o}" for s, o in panels]))


if __name__ == "__main__":
    main()
