#!/usr/bin/env python3
"""Illustrate all-to-all experimental correlations, including collision systems."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "mcmc/02_09_all_bins_input_correlations"
if os.environ.get("HIC_PYTHON_PACKAGES"):
    sys.path.insert(0, os.environ["HIC_PYTHON_PACKAGES"])
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from plot_sector_input_correlations import panel, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=float, default=.5,
                        help="Dimensionless RBF e-folding length on x_i=i/(N-1)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not np.isfinite(args.length) or args.length <= 0:
        parser.error("--length must be finite and positive")
    source = ROOT / "mcmc/02_08_sector_input_correlations"
    with (source / "bin_index.csv").open() as handle:
        bins = list(csv.DictReader(handle))
    with (source / "observable_groups.csv").open() as handle:
        groups = list(csv.DictReader(handle))
    for group in groups:
        for key in ("number", "start", "stop"):
            group[key] = int(group[key])
    n = len(bins)
    if n != 917 or [int(row["global_index"]) for row in bins] != list(range(n)):
        raise ValueError("Expected all 917 bins in saved calibration order")
    with np.load(source / "cs_sector_input_matrices.npz") as archive:
        variance = np.diag(archive["covariance"]).copy()
    if variance.shape != (n,) or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Invalid reconstructed experimental input variances")
    std = np.sqrt(variance)
    x = np.arange(n, dtype=float) / (n - 1)
    correlations = dict(cs=np.ones((n, n)),
                        rbf=np.exp(-((x[:, None] - x[None, :]) / args.length)**2))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    manifest = dict(
        scope="All 917 bins, INCLUDING cross-observable, cross-sector and cross-system pairs",
        interpretation="User-requested illustration, NOT verified as the covariance used in an existing calibration",
        cs_rho=1, rbf_length=args.length, rbf_coordinate="x_i=i/(N-1), i=0,...,916",
        rbf_kernel="exp(-((x_i-x_j)/length)^2)",
        ordering_warning="RBF distances depend on concatenated bin order, not physical distances. Reordering bins and rebuilding the kernel changes the model.",
        variance_source="Reconstructed experimental variances in likelihood coordinates from the sector study; original remote covariance hashes were not matched. Correlation maps themselves do not depend on these positive variances.",
        n_bins=n, systems={"Au-Au-200": 791, "d-Au-200": 126}, validation={},
        color_scale=dict(colormap="coolwarm", minimum=0, maximum=1,
                         negative_correlations_allowed=False),
        sources={str(p.relative_to(ROOT)): sha(p) for p in
                 [source / "bin_index.csv", source / "observable_groups.csv",
                  source / "cs_sector_input_matrices.npz", Path(__file__),
                  ROOT / "mcmc/scripts/plot_sector_input_correlations.py"]})
    with threadpool_limits(limits=2):
        for model, correlation in correlations.items():
            covariance = np.outer(std, std) * correlation
            np.testing.assert_allclose(correlation, correlation.T, atol=1e-14)
            np.testing.assert_allclose(np.diag(correlation), 1, atol=1e-14)
            np.testing.assert_allclose(np.diag(covariance), variance, rtol=1e-14)
            np.testing.assert_allclose(
                covariance / np.outer(std, std), correlation, atol=1e-14)
            if not np.isfinite(correlation).all() or np.any(correlation < 0) or np.any(correlation > 1):
                raise ValueError("Invalid correlation coefficients")
            eigenvalues = np.linalg.eigvalsh(correlation)
            tolerance = n * np.finfo(float).eps * max(abs(eigenvalues).max(), 1)
            if eigenvalues.min() < -tolerance:
                raise ValueError("Correlation is not positive semidefinite")
            if not np.all(correlation[:791, 791:] > 0):
                raise ValueError("Expected nonzero cross-system correlations")
            if model == "rbf":
                np.testing.assert_allclose(correlation[0, -1], np.exp(-1 / args.length**2))
                np.testing.assert_allclose(correlation[0, 458], np.exp(-(.5 / args.length)**2))
            manifest["validation"][model] = dict(
                minimum_eigenvalue=float(eigenvalues.min()), psd_tolerance=float(tolerance),
                numerical_rank=int(np.sum(eigenvalues > tolerance)),
                first_last_correlation=float(correlation[0, -1]),
                first_middle_correlation=float(correlation[0, 458]),
                minimum_cross_system_correlation=float(correlation[:791, 791:].min()))
            np.savez_compressed(args.output / f"{model}_all_bins_input_matrices.npz",
                                correlation=correlation, covariance=covariance, coordinate=x,
                                global_index=np.arange(n))
    for name in ("bin_index.csv", "observable_groups.csv"):
        (args.output / name).write_bytes((source / name).read_bytes())
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    fig.subplots_adjust(left=.045, right=.91, top=.84, bottom=.16, wspace=.16)
    for ax, model in zip(axes, correlations):
        image = panel(ax, correlations[model], groups)
        title = r"Compound symmetry: $\rho=1$ for every pair" if model == "cs" else (
            rf"RBF: $\ell={args.length:g}$, $x_i=i/916$" + "\n" +
            r"$R_{ij}=\exp[-((x_i-x_j)/\ell)^2]$")
        ax.set_title(title, fontsize=15, pad=16)
    fig.colorbar(image, cax=fig.add_axes([.935, .25, .012, .51]),
                 label="Experimental Pearson correlation", ticks=[0, .25, .5, .75, 1])
    fig.suptitle("All-to-all correlation model: all 917 experimental bins", fontsize=20, y=.975)
    fig.text(.5, .935, "Includes Au-Au / d-Au correlations | Black lines: system boundary "
             "(NOT a correlation cut) | Thin lines: observable boundaries", ha="center", fontsize=11)
    fig.text(.5, .105, "CS: every standardized experimental error moves together. "
             "RBF: correlation decays with dimensionless distance in the concatenated bin list.",
             ha="center", fontsize=11)
    fig.text(.5, .07, "RBF uses bin order, not physical distances. Changing that order and "
             "rebuilding the kernel changes the correlation model.", ha="center", fontsize=11)
    fig.text(.5, .035, "Illustration of the requested model; NOT certified as the input "
             "used in the original calibrations. GP covariance is not included.",
             ha="center", fontsize=11, color="#944b35")
    for extension in ("pdf", "png"):
        fig.savefig(args.output / f"cs_vs_rbf_all_bins_input_correlation.{extension}", dpi=300)
    plt.close(fig)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest["validation"], indent=2))
    print("Saved full all-to-all maps and numerical matrices to", args.output)


if __name__ == "__main__":
    main()
