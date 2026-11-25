#!/usr/bin/env python3
"""F28-corner: full 20x20 corner (Figs. 28+29 in one), baseline vs one model (DESIGN_SYSTEM 2.01).

Baseline: green filled 2D density + green 1D. Model: 68/90% contours + 1D in model colour.
Axes fixed to the prior ranges (Table I). Diagonal titles: med^{+}_{-} for both.
Usage: python fig_corner.py --proposal sector --model cs_sector --outdir <...>/01_F28-corner
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter

import common
import style

BINS, SMOOTH = 40, 1.2


def levels(density, probs=(0.90, 0.68)):
    o = np.sort(density.ravel())[::-1]
    mass = np.cumsum(o) / o.sum()
    lv = np.unique([o[min(np.searchsorted(mass, p), len(o) - 1)] for p in probs])
    return lv[(lv > density.min()) & (lv < density.max())]


def hist2d(x, y, rx, ry):
    h, xe, ye = np.histogram2d(x, y, bins=BINS, range=[rx, ry])
    return gaussian_filter(h.T, SMOOTH, mode="reflect"), xe, ye


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", choices=common.PROPOSALS, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    if a.model not in common.PROPOSALS[a.proposal][1:]:
        ap.error("model not in proposal")
    style.apply()
    base, mod = common.load("baseline"), common.load(a.model)
    spec = common.MODELS[a.model]
    ranges = common.prior_ranges()
    n = 20
    t = (1 - style.CI) / 2
    qb = np.quantile(base, [t, .5, 1 - t], axis=0)
    qm = np.quantile(mod, [t, .5, 1 - t], axis=0)
    cmap = LinearSegmentedColormap.from_list("g", ["white", "#e3f1df", "#a9d69c", style.BASELINE])

    fig, axes = plt.subplots(n, n, figsize=(15, 15))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.97, wspace=0.06, hspace=0.06)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            ax.set_xlim(ranges[j])
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.tick_params(labelsize=6, length=2, pad=1,
                           labelbottom=(i == n - 1), labelleft=(j == 0 and i > 0))
            if i == j:
                edges = np.linspace(*ranges[i], BINS + 1)
                hb, _ = np.histogram(base[:, i], edges, density=True)
                hm, _ = np.histogram(mod[:, i], edges, density=True)
                ax.stairs(hb, edges, color=style.BASELINE, lw=1.1, fill=True, alpha=0.35)
                ax.stairs(hb, edges, color=style.BASELINE, lw=1.1)
                ax.stairs(hm, edges, color=spec["color"], lw=1.2)
                ax.set_ylim(0, 1.15 * max(hb.max(), hm.max()))
                ax.set_yticks([])
                for q, c in ((qb, style.BASELINE), (qm, spec["color"])):
                    ax.axvline(q[1, i], color=c, lw=0.8)
                    for k in (0, 2):
                        ax.axvline(q[k, i], color=c, lw=0.7, ls=(0, (2, 1.5)))
                lo, md, hi = qb[:, i]
                lo2, md2, hi2 = qm[:, i]
                ax.text(0.5, 1.30, rf"${md:.2f}^{{+{hi-md:.2f}}}_{{-{md-lo:.2f}}}$", color=style.BASELINE,
                        transform=ax.transAxes, ha="center", fontsize=7)
                ax.text(0.5, 1.05, rf"${md2:.2f}^{{+{hi2-md2:.2f}}}_{{-{md2-lo2:.2f}}}$", color=spec["color"],
                        transform=ax.transAxes, ha="center", fontsize=7)
            else:
                ax.set_ylim(ranges[i])
                ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
                d, xe, ye = hist2d(base[:, j], base[:, i], ranges[j], ranges[i])
                ax.pcolormesh(xe, ye, d, cmap=cmap, shading="flat", rasterized=True)
                xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
                lv = levels(d)
                if lv.size:
                    ax.contour(xc, yc, d, levels=lv, colors=style.BASELINE, linewidths=0.6)
                dm, _, _ = hist2d(mod[:, j], mod[:, i], ranges[j], ranges[i])
                lv = levels(dm)
                if lv.size:
                    ax.contour(xc, yc, dm, levels=lv, colors=spec["color"], linewidths=0.8)
            if i == n - 1:
                ax.set_xlabel(common.LABELS[j], fontsize=10, rotation=90)
            if j == 0 and i > 0:
                ax.set_ylabel(common.LABELS[i], fontsize=10, rotation=0, ha="right", va="center")
    fig.legend(handles=[Line2D([], [], color=style.BASELINE, lw=3, label="baseline (diag): density, 68/90%"),
                        Line2D([], [], color=spec["color"], lw=2, label=f"{spec['label']}: 68/90% contours")],
               loc="upper right", bbox_to_anchor=(0.97, 0.90), frameon=False, fontsize=16)
    fig.text(0.62, 0.78, "Diagonal: median and 90% C.I. (solid / dashed)\nTitles: baseline (top), model (bottom)",
             fontsize=12, color="#555555")
    stem = f"F28-corner__{a.proposal}__baseline-vs-{common.SHORT[a.model]}"
    style.save(fig, a.outdir, stem)
    names, _, _ = common.design()
    common.write_manifest(a.outdir, stem, dict(
        common.provenance(["baseline", a.model]), bins=BINS, smooth=SMOOTH,
        posterior={m: {names[k]: dict(lo=float(q[0, k]), median=float(q[1, k]), hi=float(q[2, k]))
                       for k in range(20)} for m, q in (("baseline", qb), (a.model, qm))}))


if __name__ == "__main__":
    main()
