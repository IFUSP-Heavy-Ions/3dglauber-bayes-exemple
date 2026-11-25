"""Per-parameter shift and width of each posterior relative to the baseline, in units of sigma_B.

shift_j = (median_j - median_j^baseline) / sigma_B,j      sigma_B = baseline posterior std
width_j = std_j / sigma_B,j

Same definition as the 1.7 sigma quoted for the CS sector on the poster.

Run on the server from the project root:
    .venv/bin/python study_nongaussian_systematics/posterior/shift_summary.py
Output: mcmc/02_11_nongaussian_nuisance_detector_split/shift_summary.{csv,png,pdf}
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mcmc" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_posterior_corner import DESIGN, LABELS, load_samples  # noqa: E402

OUT = ROOT / "mcmc/02_11_nongaussian_nuisance_detector_split"
CHAINS = {
    "baseline": ROOT / "mcmc/01_01_baseline_reproduction/chain-idf-0.hdf",
    "cs": ROOT / "mcmc/02_04_cs_sector/chain-idf-0-cs-observable-region-rho-1-a77902df73a5.hdf",
    "nuis_prev": next((ROOT / "mcmc/02_10_nongaussian_nuisance").glob("chain-idf-0-nongauss-nuisance-*.hdf")),
    "nuis": OUT / "chain-idf-0-nongauss-nuisance-detector-split.hdf",
}
MODELS = [("cs", "CS sector ($\\rho=1$)", "#477fb5"),
          ("nuis_prev", "Nuisance, global normalization", "#8c6bb1"),
          ("nuis", "Nuisance, TPC/FTPC normalizations", "#2a9d5c")]


def main():
    with DESIGN.open() as handle:
        names = [row["param"] for row in csv.DictReader(handle)]
    samples = {k: load_samples(p, 0, 50, 200000)[0][:, :20] for k, p in CHAINS.items()}
    base = samples["baseline"]
    med_b, sig_b = np.median(base, 0), base.std(0)
    rows, shift, width = [], {}, {}
    for key, _, _ in MODELS:
        s = samples[key]
        shift[key] = (np.median(s, 0) - med_b) / sig_b
        width[key] = s.std(0) / sig_b
    for j, name in enumerate(names):
        row = {"parameter": name, "baseline_median": med_b[j], "sigma_B": sig_b[j]}
        for key, _, _ in MODELS:
            row[f"{key}_shift_sigmaB"] = shift[key][j]
            row[f"{key}_width_ratio"] = width[key][j]
        rows.append(row)
    with (OUT / "shift_summary.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    labels = list(LABELS[:20])
    y = np.arange(len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
    h = 0.27
    for i, (key, lab, col) in enumerate(MODELS):
        off = (i - 1) * h
        a1.barh(y + off, shift[key], h, color=col, label=lab)
        a2.barh(y + off, width[key] - 1, h, left=1, color=col)
    for ax in (a1, a2):
        ax.grid(axis="x", color="0.9"); ax.set_axisbelow(True)
    a1.axvline(0, color="k", lw=0.8)
    for v in (-1, 1):
        a1.axvline(v, color="0.5", ls=":", lw=0.8)
    a2.axvline(1, color="k", lw=0.8)
    a1.set_yticks(y, labels); a1.invert_yaxis()
    a1.set_xlabel(r"shift of median $/\ \sigma_B$")
    a2.set_xlabel(r"posterior width $/\ \sigma_B$")
    a1.legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Shift and width relative to the baseline (diagonal) posterior")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"shift_summary.{ext}", dpi=200)
    for key, lab, _ in MODELS:
        j = np.argmax(np.abs(shift[key]))
        print(f"{lab:40s} max|shift| = {abs(shift[key][j]):.2f} sigma_B ({names[j]}); "
              f"median|shift| = {np.median(np.abs(shift[key])):.2f}; "
              f"width ratio range {width[key].min():.2f}-{width[key].max():.2f}")


if __name__ == "__main__":
    main()
