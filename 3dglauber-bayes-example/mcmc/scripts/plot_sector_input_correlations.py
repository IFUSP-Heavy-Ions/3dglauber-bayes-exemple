#!/usr/bin/env python3
"""Reconstruct experimental input correlations from saved sector-chain metadata."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "mcmc/02_08_sector_input_correlations"
sys.path.insert(0, str(ROOT))
if os.environ.get("HIC_PYTHON_PACKAGES"):
    sys.path.insert(0, os.environ["HIC_PYTHON_PACKAGES"])
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib-cache"))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits

from experimental_sector_covariance import (
    bin_centers, covariance_signature, likelihood_std, observable_axis,
    rapidity_region_labels, resolve_length, sector_covariance, validate_overrides,
    validate_regions,
)

STUDIES = {
    "cs": "02_04_cs_sector",
    "rbf": "02_05_rbf_sector",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct(config, model):
    study = ROOT / "mcmc" / STUDIES[model]
    saved = json.loads((study / "study_manifest.json").read_text())
    chain = study / saved["chain"]
    with h5py.File(chain, "r") as handle:
        attrs = dict(handle.attrs)
        datasets = list(handle.keys())
    if attrs != saved["chain_attributes"]:
        raise ValueError(f"Saved manifest and chain metadata disagree: {chain}")
    expected = "compound_symmetry" if model == "cs" else "rbf"
    if attrs["experimental_covariance_model"] != expected:
        raise ValueError(f"Unexpected covariance model: {chain}")
    if attrs["experimental_covariance_scope"] != "observable_rapidity_region":
        raise ValueError(f"Unexpected covariance scope: {chain}")
    regions = json.loads(attrs["experimental_rapidity_regions"])
    lengths = json.loads(attrs.get("experimental_rbf_lengths", "{}"))
    overrides = json.loads(attrs.get("experimental_rbf_overrides", "{}"))
    rho = float(attrs.get("experimental_cs_rho", 1.0))
    bins = {system: config.obs_cent_list[system] for system in config.system_strs}
    validate_regions(regions, bins)
    validate_overrides(overrides, bins)
    matrices, slices, records, groups, sources = {}, {}, [], [], set()
    offset = 0
    for system in config.system_strs:
        slices[system] = []
        sigmas = []
        for obs in config.active_obs_list[system]:
            edges = np.asarray(bins[system][obs], dtype=float)
            n = len(edges)
            source = ROOT / config.dir_obs_exp / system / config.expt_label_dict[obs[-4:]] / (obs + ".dat")
            sources.add(source)
            frame = pd.read_csv(source, sep=" ", skiprows=2, escapechar="#").iloc[:n]
            mean = frame["val"].to_numpy(float)
            variance = frame["err"].to_numpy(float)
            if len(frame) != n or not np.isfinite(variance).all() or np.any(variance < 0):
                raise ValueError(f"Invalid experimental variance or bin count: {source}")
            std = likelihood_std(obs, mean, np.sqrt(variance), config.transform_multiplicities)
            if np.any(std <= 0):
                raise ValueError(f"Pearson correlation undefined for zero variance: {source}")
            reference = pd.read_csv(study / "plots/observables" / f"{system}_{obs}.csv")
            np.testing.assert_allclose(
                reference[["bin_low", "bin_high", "experiment", "experimental_std"]].to_numpy(),
                np.column_stack((edges, mean, np.sqrt(variance))), rtol=1e-12, atol=1e-14,
                err_msg=f"Experimental inputs differ from saved study: {system}/{obs}")
            start = len(sigmas)
            slices[system].append((obs, slice(start, start + n)))
            sigmas.extend(std)
            labels = rapidity_region_labels(system, obs, edges, regions)
            groups.append(dict(number=len(groups) + 1, system=system, observable=obs,
                               start=offset + start, stop=offset + start + n))
            for k, center in enumerate(bin_centers(edges)):
                records.append(dict(global_index=offset + start + k, system=system,
                    observable=obs, observable_bin=k, bin_low=float(edges[k, 0]),
                    bin_high=float(edges[k, 1]), center=float(center),
                    coordinate=observable_axis(obs), sector=str(labels[k]),
                    measurement=float(mean[k]), experimental_std=float(np.sqrt(variance[k])),
                    likelihood_std=float(std[k])))
        matrix = sector_covariance(system, slices[system], np.array(sigmas), bins, model,
                                   rho=rho, lengths=lengths, overrides=overrides, regions=regions)
        np.testing.assert_allclose(np.diag(matrix), np.square(sigmas), rtol=1e-14)
        matrices[system] = matrix
        offset += len(sigmas)
    signature = covariance_signature(matrices, slices, bins)
    matches = signature == attrs["experimental_covariance_signature"]
    if not matches:
        warnings.warn(f"{model}: reconstructed covariance hash differs from saved chain; "
                      "exact remote input reproduction is NOT certified.", RuntimeWarning)
    covariance = block_diag(*[matrices[system] for system in config.system_strs])
    std = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std, std)
    np.testing.assert_allclose(correlation, correlation.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(correlation), 1, atol=1e-14)
    if not np.isfinite(correlation).all() or np.max(np.abs(correlation)) > 1 + 1e-12:
        raise ValueError("Invalid Pearson correlation values")
    group_ids = np.array([(r["system"], r["observable"], r["sector"]) for r in records])
    same_sector = np.all(group_ids[:, None, :] == group_ids[None, :, :], axis=2)
    np.testing.assert_array_equal(correlation[~same_sector], 0)
    if model == "cs":
        np.testing.assert_allclose(correlation[same_sector & ~np.eye(offset, dtype=bool)], rho,
                                   atol=1e-14)
    else:
        for group in groups:
            sl = slice(group["start"], group["stop"])
            local = records[sl]
            length = resolve_length(group["system"], group["observable"], lengths, overrides)
            x = np.array([r["center"] for r in local])
            labels = np.array([r["sector"] for r in local])
            expected_block = np.exp(-((x[:, None] - x[None, :]) / length)**2)
            expected_block[labels[:, None] != labels[None, :]] = 0
            np.testing.assert_allclose(correlation[sl, sl], expected_block, atol=1e-14)
    eig = np.linalg.eigvalsh(correlation)
    tolerance = offset * np.finfo(float).eps * max(abs(eig).max(), 1)
    if eig.min() < -tolerance:
        raise ValueError("Correlation is not positive semidefinite")
    audit = dict(model=model, chain=str(chain.relative_to(ROOT)), chain_datasets=datasets,
        chain_attributes=attrs, n_bins=offset, n_observables=len(groups),
        systems=[dict(system=s, n_bins=len(matrices[s])) for s in config.system_strs],
        reconstructed_signature=signature, chain_signature=attrs["experimental_covariance_signature"],
        exact_chain_signature_matches=matches,
        scope="Experimental input only; not parameter-dependent experimental + GP covariance",
        reconstruction="Native sector_covariance and likelihood_std; chain-stored regions/kernel settings; local experimental inputs verified against saved study CSVs.",
        warning="Original remote covariance is not stored in the chain. A differing signature prevents certifying exact reproduction.",
        minimum_correlation_eigenvalue=float(eig.min()), psd_tolerance=float(tolerance),
        numerical_rank=int(np.sum(eig > tolerance)),
        validated=["917-bin ordering shared between calibrations", "saved experimental bin edges/values/errors",
                   "symmetry", "unit diagonal", "Pearson bounds", "PSD",
                   "zero cross-system/observable/sector entries", "native kernel formula"],
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)},
        versions=dict(numpy=np.__version__, matplotlib=matplotlib.__version__, pandas=pd.__version__))
    audit["color_scale"] = dict(colormap="coolwarm", minimum=0, maximum=1,
                               negative_correlations_allowed=False)
    return covariance, correlation, records, groups, audit


def panel(ax, correlation, groups):
    if not np.isfinite(correlation).all() or np.any(correlation < 0):
        raise ValueError("The 0-to-1 color scale cannot display negative or nonfinite correlations")
    n = len(correlation)
    image = ax.imshow(correlation, origin="lower", interpolation="nearest",
                      extent=(-.5, n-.5, -.5, n-.5), cmap="coolwarm", vmin=0, vmax=1,
                      rasterized=True)
    ax.set_yticks(np.arange(0, n, 100))
    ax.set_xticks(np.arange(0, n, 100))
    ax.set_xlabel("Au-Au and d-Au experimental bin index")
    ax.set_ylabel("Au-Au and d-Au experimental bin index")
    for group in groups[:-1]:
        boundary = group["stop"] - .5
        system_boundary = group["system"] != groups[group["number"]]["system"]
        for draw in (ax.axhline, ax.axvline):
            draw(boundary, color="#222222" if system_boundary else "#777777",
                 lw=1.2 if system_boundary else .25, alpha=.85 if system_boundary else .3)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / "matplotlib-cache"))
    import configurations as config
    if (config.run_closure or config.pseudovalidation or config.change_exp_error
            or config.set_exp_error_to_zero or config.number_of_models_per_run != 1):
        raise ValueError("Requires real unchanged experimental errors and one model per run")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    audits, correlations, first_records, first_groups = {}, {}, None, None
    with threadpool_limits(limits=2):
        for model in STUDIES:
            covariance, correlation, records, groups, audit = reconstruct(config, model)
            if len(records) != 917:
                raise ValueError(f"Expected 917 saved calibration bins, got {len(records)}")
            if first_records is not None and records != first_records:
                raise ValueError("CS and RBF experimental bin mapping differs")
            first_records, first_groups = records, groups
            audits[model], correlations[model] = audit, correlation
            np.savez_compressed(args.output / f"{model}_sector_input_matrices.npz",
                                covariance=covariance, correlation=correlation,
                                global_index=np.arange(len(records)))
            fig, ax = plt.subplots(figsize=(15, 13))
            fig.subplots_adjust(left=.43, right=.91, bottom=.09, top=.88)
            image = panel(ax, correlation, groups)
            fig.text(.015, .91, "Observable groups and global bin ranges (inclusive)", fontsize=10)
            for i, group in enumerate(groups):
                label = (f'{group["number"]:02d} [{group["start"]:03d}-{group["stop"]-1:03d}] '
                         f'{group["system"]}: {group["observable"]}')
                fig.text(.015, .88 - i * .019, label, fontsize=7)
            ax.set_title(f"{model.upper()} sector: experimental input correlation\n"
                         f"{len(records)} x {len(records)} bins; Au-Au: 791 | d-Au: 126", pad=20)
            fig.colorbar(image, ax=ax, fraction=.035, pad=.025,
                         label=r"$R_{ij}=C_{ij}/\sqrt{C_{ii}C_{jj}}$")
            fig.text(.52, .045, "Reconstructed from chain metadata and local experimental inputs; "
                     "exact remote covariance hash NOT matched.", ha="center", fontsize=10, color="#944b35")
            for ext in ("pdf", "png"):
                fig.savefig(args.output / f"{model}_sector_input_correlation.{ext}", dpi=300)
            plt.close(fig)
            print(f"{model.upper()}: {audit['n_bins']} bins, {audit['n_observables']} observables, "
                  f"rank={audit['numerical_rank']}, chain hash match={audit['exact_chain_signature_matches']}", flush=True)
    write_csv(args.output / "bin_index.csv", first_records)
    write_csv(args.output / "observable_groups.csv", first_groups)
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    fig.subplots_adjust(left=.045, right=.91, top=.84, bottom=.13, wspace=.16)
    for ax, model in zip(axes, STUDIES):
        image = panel(ax, correlations[model], first_groups)
        subtitle = r"Within-sector $\rho=1$" if model == "cs" else r"$R_{ij}=\exp[-((x_i-x_j)/\ell)^2]$ within sector"
        ax.set_title(f"{model.upper()} sector\n{subtitle}", fontsize=15, pad=16)
    fig.colorbar(image, ax=axes, cax=fig.add_axes([.935, .23, .012, .54]),
                 label="Experimental Pearson correlation", ticks=[0, .25, .5, .75, 1])
    fig.suptitle("Experimental input correlations: all 917 calibration bins", fontsize=20, y=.975)
    fig.text(.5, .935, "Au-Au (791 bins) + d-Au (126 bins) | Black lines: system boundary | "
             "Thin lines: observable boundaries", ha="center", fontsize=11)
    fig.text(.5, .07, "Experimental covariance only; cross-system and cross-observable entries are zero. "
             "GP covariance is not included.", ha="center", fontsize=12)
    fig.text(.5, .035, "Reconstruction from chain metadata; original remote covariance not stored. "
             "Exact chain hashes NOT matched for either calibration.", ha="center", fontsize=11, color="#944b35")
    for ext in ("pdf", "png"):
        fig.savefig(args.output / f"cs_vs_rbf_sector_input_correlation.{ext}", dpi=300)
    plt.close(fig)
    audits["source_hashes"] = {str(p.relative_to(ROOT)): sha(p) for p in
        [Path(__file__), ROOT / "configurations.py", ROOT / "bins_and_cuts.py",
         ROOT / "experimental_sector_covariance.py", ROOT / "experimental_cs_covariance.py",
         ROOT / "bayes_mcmc_sector.py", ROOT / "bayes_mcmc_cs_model.py"]}
    (args.output / "manifest.json").write_text(json.dumps(audits, indent=2, allow_nan=False) + "\n")
    print("Saved full matrices, bin mapping, PNG/PDF figures and audit manifest to", args.output)


if __name__ == "__main__":
    main()
