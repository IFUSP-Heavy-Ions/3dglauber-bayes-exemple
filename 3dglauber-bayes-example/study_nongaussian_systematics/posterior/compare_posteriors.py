"""Baseline vs CS sector vs non-Gaussian nuisance: parameters and all 40 observables.

Run on the server from the project root, after the nuisance chain has finished:
    .venv/bin/python study_nongaussian_systematics/posterior/compare_posteriors.py

Reuses the saved baseline / CS-sector predictions of mcmc/02_06 and 02_04 (same 1000
posterior inputs) and computes the nuisance-chain predictions with the same emulators.
Outputs go to mcmc/02_10_nongaussian_nuisance/ (figures, CSVs, manifest).
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import dill
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mcmc" / "scripts"))
from build_sector_studies import observable_panel, physical_predictions, validate_reference  # noqa: E402
from diagnose_phase2_numerical import load_emulator_module  # noqa: E402
from plot_posterior_corner import DESIGN, LABELS, load_samples  # noqa: E402

BASELINE = ROOT / "mcmc/01_01_baseline_reproduction/chain-idf-0.hdf"
CS = ROOT / "mcmc/02_04_cs_sector/chain-idf-0-cs-observable-region-rho-1-a77902df73a5.hdf"
BASE_PRED = ROOT / "mcmc/02_06_baseline_vs_cs_sector/baseline_predictions"
CS_PRED = ROOT / "mcmc/02_04_cs_sector/plots/observables"
OUT = ROOT / "mcmc/02_10_nongaussian_nuisance"
SOURCES = ROOT / "study_nongaussian_systematics/data/inferred_sources.npz"
SYSTEM, OBS = "Au-Au-200", "v22_eta_cen_20_70_STAR"
NUIS_LABELS = [r"$\lambda_\mathrm{norm}$", r"$\lambda_\mathrm{asym}$", r"$\lambda_\mathrm{edge}$"]
MODELS = [("baseline", "Baseline (diagonal)", "#df9393", "--"),
          ("cs", "CS sector (rho=1)", "#477fb5", "-"),
          ("nuis", "Non-Gaussian nuisance", "#2a9d5c", "-")]
TAG = "3way"   # output file tag; becomes "4way" when --previous adds the earlier nuisance chain


def latex_label(name):
    """lambda_norm_TPC -> $\\lambda_\\mathrm{norm\\,TPC}$"""
    return r"$\lambda_\mathrm{" + name.replace("lambda_", "", 1).replace("_", r"\,") + "}$"
Q = [.05, .16, .5, .84, .95]
QN = ["q05", "q16", "median", "q84", "q95"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save(fig, stem):
    fig.savefig(f"{stem}.png", dpi=160); fig.savefig(f"{stem}.pdf"); plt.close(fig)


def find_nuisance_chain():
    chains = sorted(OUT.glob("chain-idf-0-nongauss-nuisance-*.hdf"))
    if len(chains) != 1:
        raise FileNotFoundError(f"expected one nuisance chain, found {chains}")
    return chains[0]


# ------------------------------------------------------------------------------ parameters
def marginals(samples, design):
    q = {k: np.quantile(v[:, :20], Q, axis=0) for k, v in samples.items()}
    fig, axes = plt.subplots(5, 4, figsize=(15, 15))
    rows = []
    for i, (ax, par) in enumerate(zip(axes.flat, design)):
        lo, hi = float(par["min"]), float(par["max"])
        for key, label, color, ls in MODELS:
            ax.hist(samples[key][:, i], bins=55, range=(lo, hi), density=True, histtype="step",
                    color=color, linestyle=ls, linewidth=1.4, label=label)
            ax.axvline(q[key][2, i], color=color, linewidth=1)
        ax.set(xlim=(lo, hi), xlabel=LABELS[i], ylabel="Density")
        width = q["baseline"][4, i] - q["baseline"][0, i]
        row = dict(parameter=par["param"])
        for key, *_ in MODELS:
            for k, name in enumerate(QN):
                row[f"{key}_{name}"] = float(q[key][k, i])
        for key in [m[0] for m in MODELS if m[0] != "baseline"]:
            row[f"{key}_shift_over_baseline_90"] = float((q[key][2, i] - q["baseline"][2, i]) / width)
            row[f"{key}_width_ratio_90"] = float((q[key][4, i] - q[key][0, i]) / width)
        rows.append(row)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Posterior marginals: baseline vs CS sector vs non-Gaussian nuisance (STAR v2(eta))")
    fig.tight_layout(rect=(0, 0, 1, .97))
    save(fig, OUT / f"posterior_marginals_{TAG}")
    write_csv(OUT / f"posterior_parameter_comparison_{TAG}.csv", rows)
    return rows


def nuisance_marginals(lam, src):
    from scipy import stats
    s, k = src["s"], 4.0 / src["skew"] ** 2
    labels = [latex_label(str(x)) for x in src["labels"]] if "labels" in src else NUIS_LABELS
    fig, axes = plt.subplots(1, len(s), figsize=(4.4 * len(s), 3.6))
    for a, ax in enumerate(axes):
        # plot in units of the prior rms, z = lambda/s (prior: mean 0, sd 1, skew 2/sqrt(k))
        z = lam[:, a] / s[a]
        bound = -np.sqrt(k[a])                                   # hard lower edge of the Gamma support
        hi = max(np.quantile(z, .999), 3.0)
        x = np.linspace(bound, hi, 600)[1:]
        prior = stats.gamma.pdf(x * np.sqrt(k[a]) + k[a], k[a]) * np.sqrt(k[a])
        h, _, _ = ax.hist(z, bins=80, range=(bound, hi), density=True, histtype="step", color="#2a9d5c",
                          linewidth=1.5, label="posterior")
        ax.plot(x, prior, color="gray", label=f"prior Gamma (k={k[a]:.2f})")
        ax.axvline(bound, color="k", ls=":", lw=1, label="support edge")
        ax.axvline(0, color="gray", ls="-", lw=.5)
        ax.set_ylim(0, 1.15 * max(h.max(), np.quantile(prior, .9)))   # k<1 prior diverges at the edge
        ax.set(xlabel=f"{labels[a]} / s   (s = {s[a]:.4f})", ylabel="Density")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, OUT / "nuisance_lambda_prior_posterior")


# ---------------------------------------------------------------------------- observables
def panel3(ax, ratio_ax, config, system, obs, preds):
    rows = observable_panel(ax, config, system, obs, preds["nuis"], "#2a9d5c")
    edges = np.asarray(config.obs_cent_list[system][obs]); centers = edges.mean(1)
    data = np.array([r["experiment"] for r in rows]); err = np.array([r["experimental_std"] for r in rows])
    starts = np.r_[0, np.flatnonzero(edges[1:, 0] > edges[:-1, 1] + 1e-8) + 1]
    for key, label, color, ls in MODELS:
        q = np.quantile(preds[key], Q, axis=0)
        for start, stop in zip(starts, np.r_[starts[1:], len(edges)]):
            slc = slice(start, stop)
            if key != "nuis":
                ax.fill_between(centers[slc], q[1, slc], q[3, slc], color=color, alpha=.22)
                ax.plot(centers[slc], q[2, slc], color=color, linestyle=ls, linewidth=1.2)
            if ratio_ax is not None:
                ratio_ax.fill_between(centers[slc], q[1, slc] / data[slc], q[3, slc] / data[slc], color=color, alpha=.22)
                ratio_ax.plot(centers[slc], q[2, slc] / data[slc], color=color, linestyle=ls, linewidth=1.1)
        for i, r in enumerate(rows):
            for k, name in enumerate(QN):
                r[f"{key}_{name}"] = float(q[k, i])
            r[f"{key}_median_over_data"] = float(q[2, i] / data[i])
    for r in rows:
        for name in QN:
            r.pop(name, None)
    ax.legend(handles=[Line2D([], [], color=c, ls=ls, label=lab) for _, lab, c, ls in MODELS]
              + [Line2D([], [], color="k", marker=".", ls="", label="Experiment")], fontsize=6)
    if ratio_ax is not None:
        ratio_ax.errorbar(centers, np.ones_like(data), yerr=err / abs(data), fmt=".", color="k", ms=3, elinewidth=.7)
        ratio_ax.axhline(1, color="gray", lw=.8)
        ratio_ax.set(xlabel=ax.get_xlabel(), ylabel="Model / data"); ax.set_xlabel("")
    return rows


TITLE = "STAR Au+Au 200 GeV, 20-70%: effect of non-Gaussian systematic structure"


def star_highlight(config, preds, lam, src):
    """The poster panel: STAR v2(eta) with the three posteriors and the inferred systematic shift."""
    edges = np.asarray(config.obs_cent_list[SYSTEM][OBS]); eta = edges.mean(1)
    data, sig_tot, sig_stat = None, src["sigma_total"], src["sigma_stat"]
    raw = [l.split() for l in (ROOT / "HIC_experimental_data/Au-Au-200/STAR" / f"{OBS}.dat").read_text().splitlines()
           if l.strip() and not l.startswith("#")]
    data = np.array(raw, float)[:, 0]
    shift = lam @ src["v_hat"].T                                   # (samples, 21) systematic offsets
    corrected = data[None, :] - shift                               # data with inferred systematics removed
    regions = [slice(0, 6), slice(6, 15), slice(15, 21)]

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
    for key, label, color, ls in MODELS:
        q = np.quantile(preds[key], Q, axis=0)
        for slc in regions:
            ax.fill_between(eta[slc], q[1, slc], q[3, slc], color=color, alpha=.25)
            ax.plot(eta[slc], q[2, slc], color=color, ls=ls, lw=1.6, label=label if slc.start == 0 else None)
            rax.plot(eta[slc], q[2, slc] / data[slc], color=color, ls=ls, lw=1.4)
            rax.fill_between(eta[slc], q[1, slc] / data[slc], q[3, slc] / data[slc], color=color, alpha=.2)
    ax.errorbar(eta, data, yerr=sig_tot, fmt="o", color="k", ms=4, lw=1, label="STAR (total error)")
    cq = np.quantile(corrected, [.16, .5, .84], axis=0)
    ax.errorbar(eta + 0.06, cq[1], yerr=[cq[1] - cq[0], cq[2] - cq[1]], fmt="D", color="#2a9d5c", mfc="white",
                ms=4, lw=1, label=r"STAR $-\sum_a\lambda_a\hat v_a$ (posterior, 68%)")
    ax.set_ylabel(r"$v_2\{2\}$"); ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_title(TITLE)
    rax.errorbar(eta, np.ones_like(data), yerr=sig_tot / data, fmt="o", color="k", ms=3, lw=.8)
    rax.axhline(1, color="gray", lw=.8); rax.set(xlabel=r"$\eta$", ylabel="model / data")
    fig.tight_layout(); save(fig, OUT / f"star_v2_eta_{TAG}")

    summary = {}
    for key, *_ in MODELS:
        med = np.median(preds[key], axis=0)
        for name, slc in (("backward_FTPC", regions[0]), ("TPC", regions[1]), ("forward_FTPC", regions[2])):
            summary[f"{key}_{name}_median_over_data"] = float(np.mean(med[slc] / data[slc]))
    summary["shift_median_backward_FTPC"] = float(np.median(shift[:, regions[0]].mean(1)))
    summary["shift_median_TPC"] = float(np.median(shift[:, regions[1]].mean(1)))
    summary["shift_median_forward_FTPC"] = float(np.median(shift[:, regions[2]].mean(1)))
    return summary


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--chain", type=Path, help="nuisance chain (default: the one in mcmc/02_10_...)")
    ap.add_argument("--out", type=Path, help="output folder (default: mcmc/02_10_nongaussian_nuisance)")
    ap.add_argument("--sources", type=Path, help="inferred_sources.npz used by the chain (default: original study)")
    ap.add_argument("--label", help="legend label of this nuisance chain")
    ap.add_argument("--previous", type=Path,
                    help="folder of an earlier nuisance comparison (chain + nuisance_predictions_*.npz) to add as a 4th model")
    ap.add_argument("--previous-label", default="Nuisance, global normalization")
    args = ap.parse_args()
    global OUT, SOURCES, TAG, TITLE
    if args.out:
        OUT = args.out
    if args.sources:
        SOURCES = args.sources.resolve()
    if args.label:
        MODELS[-1] = ("nuis", args.label, *MODELS[-1][2:])
    prev_dir = args.previous
    if prev_dir:
        MODELS.insert(2, ("nuis_prev", args.previous_label, "#8c6bb1", ":"))
        TAG = "4way"
        TITLE = "STAR Au+Au 200 GeV, 20-70%: global vs per-detector systematic normalization"
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "observables").mkdir(exist_ok=True)
    with DESIGN.open() as handle:
        design = list(csv.DictReader(handle))
    nchain = args.chain or find_nuisance_chain()
    samples = {"baseline": load_samples(BASELINE, 0, 50, 200000)[0],
               "cs": load_samples(CS, 0, 50, 200000)[0]}
    nuis, nmeta = load_samples(nchain, 0, 50, 200000)
    samples["nuis"] = nuis
    if prev_dir:
        prev_chain = sorted(prev_dir.glob("chain-idf-0-nongauss-nuisance-*.hdf"))
        assert len(prev_chain) == 1, prev_chain
        samples["nuis_prev"] = load_samples(prev_chain[0], 0, 50, 200000)[0]
    src = dict(np.load(SOURCES))
    rows = marginals(samples, design)
    nuisance_marginals(nuis[:, 20:], src)

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(nuis), size=1000, replace=False)
    theta_n, lam_n = nuis[pick, :20], nuis[pick, 20:]
    np.savez_compressed(OUT / "nuisance_posterior_inputs.npz", theta=theta_n, lam=lam_n, index=pick)

    config = __import__("configurations")
    preds = {}
    with threadpool_limits(limits=1):
        module = load_emulator_module()
        for system in config.system_strs:
            with (ROOT / f"emulator/emulator-{system}-idf-0-npc-10.dill").open("rb") as handle:
                emu = dill.load(handle)
            emu.__class__ = module.Emulator
            validate_reference(emu, system)
            preds[system] = {
                "baseline": dict(np.load(BASE_PRED / f"predictions_{system}.npz")),
                "cs": dict(np.load(CS_PRED / f"predictions_{system}.npz")),
                "nuis": physical_predictions(emu, theta_n, config.active_obs_list[system]),
            }
            if prev_dir:
                preds[system]["nuis_prev"] = dict(np.load(prev_dir / f"nuisance_predictions_{system}.npz"))
            np.savez_compressed(OUT / f"nuisance_predictions_{system}.npz", **preds[system]["nuis"])

    all_rows = []
    for system in config.system_strs:
        observables = config.active_obs_list[system]
        for obs in observables:
            p = {k: preds[system][k][obs] for k, *_ in MODELS}
            fig, (ax, rax) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            r = panel3(ax, rax, config, system, obs, p)
            fig.tight_layout(); save(fig, OUT / "observables" / f"{system}_{obs}")
            all_rows.extend(r)
        with PdfPages(OUT / "observables" / f"posterior_comparison_{TAG}_{system}.pdf") as pdf:
            for page, start in enumerate(range(0, len(observables), 9), start=1):
                fig, axes = plt.subplots(3, 3, figsize=(16, 12))
                sel = observables[start:start + 9]
                for ax, obs in zip(axes.flat, sel):
                    panel3(ax, None, config, system, obs, {k: preds[system][k][obs] for k, *_ in MODELS})
                for ax in axes.flat[len(sel):]:
                    ax.set_visible(False)
                fig.suptitle(f"{system}: baseline vs CS sector vs non-Gaussian nuisance (posterior means, 68%)")
                fig.tight_layout(rect=(0, 0, 1, .96))
                fig.savefig(OUT / "observables" / f"posterior_comparison_{TAG}_{system}_page{page:02d}.png", dpi=150)
                pdf.savefig(fig); plt.close(fig)
    write_csv(OUT / f"posterior_predictions_comparison_{TAG}.csv", all_rows)

    star = star_highlight(config, {k: preds[SYSTEM][k][OBS] for k, *_ in MODELS}, lam_n, src)
    manifest = dict(
        baseline=str(BASELINE.relative_to(ROOT)), cs_sector=str(CS.relative_to(ROOT)),
        nuisance=dict(path=str(nchain), **nmeta,
                      attributes={k: (v.item() if isinstance(v, np.generic) else v)
                                  for k, v in h5py.File(nchain, "r").attrs.items()}),
        sources=dict(path=str(SOURCES.relative_to(ROOT)), sha256=sha(SOURCES)),
        previous=str(prev_dir) if prev_dir else None, models=[m[:2] for m in MODELS],
        n_prediction_samples=1000, seed=args.seed, star_v2_eta_summary=star,
        bands="Posterior variation of emulator means (68%); no emulator or experimental noise draws",
        versions=dict(python=platform.python_version(), numpy=np.__version__),
        script_sha256=sha(__file__),
    )
    (OUT / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(star, indent=1))
    largest = sorted(rows, key=lambda r: -abs(r["nuis_shift_over_baseline_90"]))[:6]
    for r in largest:
        print(f"{r['parameter']:45s} nuis {r['nuis_shift_over_baseline_90']:+.2f}  cs {r['cs_shift_over_baseline_90']:+.2f}  "
              f"width nuis {r['nuis_width_ratio_90']:.2f}")


if __name__ == "__main__":
    main()
