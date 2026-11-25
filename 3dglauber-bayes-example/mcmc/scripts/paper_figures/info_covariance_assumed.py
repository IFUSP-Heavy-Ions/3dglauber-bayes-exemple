#!/usr/bin/env python3
"""I1-covariance for proposals with an ASSUMED experimental covariance (1 global, 2 sector) (DESIGN_SYSTEM 2.08).

Rebuilds the experimental covariance of each model of the proposal with the code that built the chains, with the
parameters read from the chain's own HDF attributes (rho, scope, lengths, rapidity regions, acceptance windows), and
writes everything the I1 article quotes:
  - manifest .json: parameters, per-observable table (bins, axis, coordinates, regions, nearest-neighbour and mean
    correlations), per-system structure (eigenvalues, rank, mean correlation), kernel values, observable x observable
    mean correlation (RBF);
  - full matrices per model: <stem>__<model>__matrix.npz (cov, corr, sigma, labels) and __matrix_cov.csv (917 x 917);
  - illustration (1x3): (a) the kernel; (b) bin coordinates (global) or sector sizes (sector); (c) mean correlation
    between observables (global RBF) or inside each sector (sector).
Covariance space: the likelihood's (sigma_i = likelihood_std of bin_index.csv; multiplicities in log).
Usage: python info_covariance_assumed.py --proposal global --outdir <...>/08_I1-covariance
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common
import emu
import style

EXP = {"PHOB": "PHOBOS", "PHEN": "PHENIX", "STAR": "STAR", "BRAH": "BRAHMS"}


def module(name):
    sys.path.insert(0, str(common.PROJECT))
    spec = importlib.util.spec_from_file_location(name, common.PROJECT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chain_attrs(model):
    with h5py.File(common.MODELS[model]["path"], "r") as f:
        out = {}
        for k, v in f.attrs.items():
            v = v.decode() if isinstance(v, bytes) else v
            try:
                out[k] = json.loads(v) if isinstance(v, str) and v[:1] in "{[" else (v.item() if hasattr(v, "item") else v)
            except json.JSONDecodeError:
                out[k] = v
        out["_shape_chain"] = list(f["chain"].shape)
    return out


def groups(rows):
    out = []
    for i, r in enumerate(rows):
        key = (r["system"], r["observable"])
        if not out or out[-1][0] != key:
            out.append([key, i, i + 1])
        else:
            out[-1][2] = i + 1
    return out


def build(model, attrs, rows):
    """Full 917 x 917 covariance in likelihood space, block-diagonal in systems."""
    n = len(rows)
    std = np.array([float(r["likelihood_std"]) for r in rows])
    cov = np.zeros((n, n))
    grp = groups(rows)
    for system in dict.fromkeys(r["system"] for r in rows):
        idx = np.array([i for i, r in enumerate(rows) if r["system"] == system])
        start = idx[0]
        sl = [(obs, slice(a - start, b - start)) for (s, obs), a, b in grp if s == system]
        bins = {system: {obs: np.array([[float(rows[start + k]["bin_low"]), float(rows[start + k]["bin_high"])]
                                        for k in range(s.start, s.stop)]) for obs, s in sl}}
        sig = std[idx]
        if model == "cs_global":
            block = module("experimental_cs_covariance").compound_symmetry_covariance(sig, rho=attrs["experimental_cs_rho"])
        elif model == "rbf_global":
            g = module("experimental_global_covariance")
            block = g.global_rbf_covariance(system, sl, sig, bins, attrs["experimental_rbf_lengths"],
                                            attrs["experimental_rbf_acceptance"])
        elif model in ("cs_sector", "rbf_sector"):
            m = module("experimental_sector_covariance")
            block = m.sector_covariance(system, sl, sig, bins, "cs" if model == "cs_sector" else "rbf",
                                        rho=attrs.get("experimental_cs_rho", 1.0),
                                        lengths=attrs.get("experimental_rbf_lengths"),
                                        overrides=attrs.get("experimental_rbf_overrides", {}),
                                        regions=attrs.get("experimental_rapidity_regions"))
        else:
            raise ValueError(model)
        cov[np.ix_(idx, idx)] = block
    return cov, std


def describe(rows, attrs_by_model):
    """Per-observable rows: experiment, bins, axis, coordinates, regions."""
    g_mod = module("experimental_global_covariance")
    s_mod = module("experimental_sector_covariance")
    acc = next((a["experimental_rbf_acceptance"] for a in attrs_by_model.values() if "experimental_rbf_acceptance" in a), None)
    regions = next((a["experimental_rapidity_regions"] for a in attrs_by_model.values() if "experimental_rapidity_regions" in a), None)
    out = []
    for (system, obs), a, b in groups(rows):
        edges = np.array([[float(r["bin_low"]), float(r["bin_high"])] for r in rows[a:b]])
        centers = edges.mean(1)
        axis = s_mod.observable_axis(obs)
        rec = dict(system=system, observable=obs, experiment=EXP[obs.rsplit("_", 1)[1]], first=a, n=b - a, axis=axis,
                   x_min=float(centers.min()), x_max=float(centers.max()),
                   spacing=float(np.median(np.diff(np.sort(centers)))) if b - a > 1 else None,
                   sigma_min=float(min(float(r["likelihood_std"]) for r in rows[a:b])),
                   sigma_max=float(max(float(r["likelihood_std"]) for r in rows[a:b])))
        if acc is not None:
            c = g_mod.bin_coordinates(system, obs, edges, acc)
            rec["coord_pt"] = sorted(set(np.round(c[:, 0], 4))) if axis != "pt" else None
            rec["coord_eta"] = sorted(set(np.round(c[:, 1], 4))) if axis != "eta" else None
            rec["coord_cent"] = sorted(set(np.round(c[:, 2], 4))) if axis != "centrality" else None
            rec["windows"] = {ax: acc[system][obs][ax]["window"] for ax in ("pt", "eta") if ax in acc[system].get(obs, {})}
        if regions is not None:
            labels = s_mod.rapidity_region_labels(system, obs, edges, regions)
            spec = regions.get(system, {}).get(obs)
            # A sector is one LABEL (sector_covariance groups by np.unique(labels)): both "lateral" sides of an
            # observable form ONE sector. sizes = bins per label; layout = contiguous runs, for the text.
            sizes = [[str(lab), int((labels == lab).sum())] for lab in dict.fromkeys(labels)]
            layout, cur = [], None
            for lab in labels:
                if lab != cur:
                    layout.append([str(lab), 0]); cur = lab
                layout[-1][1] += 1
            rec["regions"] = dict(boundaries=spec["boundaries"] if spec else None, labels=spec["labels"] if spec else None,
                                  sizes=sizes, layout=layout)
        out.append(rec)
    return out


def stats(cov, std, rows, obs_rows):
    corr = cov / np.outer(std, std)
    per_obs = []
    for o in obs_rows:
        a, n = o["first"], o["n"]
        idx = np.arange(a, a + n)
        same = np.array([i for i, r in enumerate(rows) if r["system"] == o["system"]])
        others = np.setdiff1d(same, idx)
        sub = corr[np.ix_(idx, idx)]
        off = sub[~np.eye(n, dtype=bool)] if n > 1 else np.array([np.nan])
        nn = [corr[i, i + 1] for i in idx[:-1]] if n > 1 else [np.nan]
        per_obs.append(dict(within_mean=float(np.nanmean(off)), within_min=float(np.nanmin(off)),
                            neighbour_mean=float(np.nanmean(nn)),
                            cross_mean=float(corr[np.ix_(idx, others)].mean()) if len(others) else None))
    per_sys = {}
    for system in dict.fromkeys(r["system"] for r in rows):
        idx = np.array([i for i, r in enumerate(rows) if r["system"] == system])
        sub_c, sub_r = cov[np.ix_(idx, idx)], corr[np.ix_(idx, idx)]
        ev = np.linalg.eigvalsh(sub_c)[::-1]
        evr = np.linalg.eigvalsh(sub_r)[::-1]
        off = sub_r[~np.eye(len(idx), dtype=bool)]
        per_sys[system] = dict(n_bins=int(len(idx)), corr_mean=float(off.mean()), frac_above_0p5=float((off > 0.5).mean()),
                               frac_exact_zero=float((np.abs(off) < 1e-12).mean()),
                               rank_corr=int((evr > 1e-8 * evr[0]).sum()), min_eig_corr=float(evr[-1]),
                               top_eig_frac_cov=[float(v) for v in ev[:3] / ev.sum()],
                               top_eig_frac_corr=[float(v) for v in evr[:3] / evr.sum()])
    return corr, per_obs, per_sys


def obs_matrix(corr, obs_rows):
    k = len(obs_rows)
    M = np.full((k, k), np.nan)
    for i, a in enumerate(obs_rows):
        for j, b in enumerate(obs_rows):
            if a["system"] != b["system"]:
                M[i, j] = 0.0
                continue
            ia = np.arange(a["first"], a["first"] + a["n"]); ib = np.arange(b["first"], b["first"] + b["n"])
            sub = corr[np.ix_(ia, ib)]
            M[i, j] = sub[~np.eye(len(ia), dtype=bool)].mean() if i == j and len(ia) > 1 else (1.0 if i == j else sub.mean())
    return M


def short(o):
    obs = o["observable"]
    sysn = "AuAu" if o["system"].startswith("Au") else "dAu"
    return f"{sysn} {obs}"


def illustration(proposal, models, attrs, obs_rows, results, outdir, stem):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1, 1.1, 1.25]))
    rbf = [m for m in models if m.startswith("rbf")][0]
    L = attrs[rbf]["experimental_rbf_lengths"]
    # (a) kernel: one curve in dx/L (same shape on every axis); adjacent-bin spacings marked per axis
    ax = axes[0]
    dx = np.linspace(0, 3, 300)
    ax.plot(dx, np.exp(-dx**2), color=style.RBF, ls=style.DASH, lw=2.5,
            label=r"RBF: $R=\exp[-(\Delta x/L)^2]$ (per axis)")
    ax.axhline(1, color=style.CS, ls=style.DASH, lw=2.5, label=r"CS: $R=\rho=1$ inside the block")
    marks = {"pt": ("^", rf"$p_T$, $L$ = {L['pt']:g} GeV/c"), "eta": ("s", rf"$\eta$, $L$ = {L['eta']:g}"),
             "centrality": ("o", rf"centrality, $L$ = {L['centrality']:g} p.p.")}
    for axis, (mk, lab) in marks.items():
        r = np.array([o["spacing"] / L[axis] for o in obs_rows if o["spacing"] and o["axis"] == axis])
        if r.size:
            ax.plot(r, np.exp(-r**2), mk, ms=6, color="black", alpha=0.6, ls="",
                    label=f"adjacent bins, {lab}")
    ax.set_xlabel(r"$\Delta x / L$")
    ax.set_ylabel(r"$R_{ij}$")
    ax.set_ylim(-0.02, 1.12)
    ax.set_title("(a) the rule: kernel vs distance")
    ax.legend(fontsize=9, loc="lower left", frameon=True, framealpha=0.9)
    # (b)
    ax = axes[1]
    if proposal == "global":
        fam = {"dNdeta": ("o", "#555555"), "v22_eta": ("s", "#999999"), "v22_pt": ("^", "black"),
               "v22_int": ("D", "#333333"), "v32_int": ("P", "#333333"), "meanpT": ("X", "#777777")}
        g_mod = module("experimental_global_covariance")
        rows = emu.bins()
        acc = attrs["rbf_global"]["experimental_rbf_acceptance"]
        seen = set()
        for o in obs_rows:
            if o["system"] != "Au-Au-200":
                continue
            edges = np.array([[float(r["bin_low"]), float(r["bin_high"])] for r in rows[o["first"]:o["first"] + o["n"]]])
            c = g_mod.bin_coordinates(o["system"], o["observable"], edges, acc)
            key = next(k for k in fam if o["observable"].startswith(k))
            mk, col = fam[key]
            ax.scatter(c[:, 1], c[:, 0], s=14, marker=mk, color=col, alpha=0.6,
                       label=key if key not in seen else None)
            seen.add(key)
        ax.set_xlabel(r"$\eta$ coordinate")
        ax.set_ylabel(r"$p_T$ coordinate [GeV/c]")
        ax.set_title("(b) Au-Au bins in the common space (pT, $\\eta$)")
        ax.legend(fontsize=9, loc="upper right", frameon=True, framealpha=0.9, title="observable family", title_fontsize=9)
    else:
        sizes = []
        for o in obs_rows:
            for lab, n in o["regions"]["sizes"]:
                sizes.append(n)
        ax.hist(sizes, bins=np.arange(0.5, max(sizes) + 1.5, 1), color="#777777")
        ax.set_xlabel("bins per sector (observable x region label)")
        ax.set_ylabel("number of sectors")
        ax.set_title(f"(b) {len(sizes)} sectors, sizes")
    # (c)
    ax = axes[2]
    if proposal == "global":
        M = np.array(results["rbf_global"]["obs_matrix"])
        im = ax.imshow(np.clip(M, 0, 1), cmap="cividis", vmin=0, vmax=1, interpolation="nearest")
        k_au = sum(1 for o in obs_rows if o["system"] == "Au-Au-200")
        ax.axhline(k_au - 0.5, color="white", lw=1.5); ax.axvline(k_au - 0.5, color="white", lw=1.5)
        ax.set_xlabel("observable (likelihood order)")
        ax.set_ylabel("observable (likelihood order)")
        ax.set_title("(c) RBF global: mean correlation between observables")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$\langle R_{ij}\rangle$ (diagonal: inside the observable)")
    else:
        x = np.arange(len(obs_rows))
        for m, off in (("cs_sector", -0.2), ("rbf_sector", 0.2)):
            v = [p["within_mean"] for p in results[m]["per_observable"]]
            ax.bar(x + off, v, width=0.4, color=common.MODELS[m]["color"], label=common.MODELS[m]["label"])
        ax.set_xlabel("observable (likelihood order)")
        ax.set_ylabel(r"mean $R_{ij}$ inside the observable (all pairs)")
        ax.set_ylim(0, 1.05)
        ax.set_title("(c) correlation inside each observable")
        ax.legend(fontsize=9, loc="lower left", frameon=True, framealpha=0.9)
    fig.tight_layout()
    style.save(fig, outdir, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", choices=["global", "sector"], required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    style.apply()
    rows = emu.bins()
    models = common.PROPOSALS[a.proposal][1:]
    attrs = {m: chain_attrs(m) for m in models}
    obs_rows = describe(rows, attrs)
    stem = f"I1-covariance__{a.proposal}__{'-'.join(common.SHORT[m] for m in models)}"
    results = {}
    a.outdir.mkdir(parents=True, exist_ok=True)
    for m in models:
        cov, std = build(m, attrs[m], rows)
        assert np.allclose(np.diag(cov), std**2), m
        corr, per_obs, per_sys = stats(cov, std, rows, obs_rows)
        res = dict(per_observable=per_obs, per_system=per_sys, chain=str(common.MODELS[m]["path"].relative_to(common.PROJECT)),
                   chain_shape=attrs[m]["_shape_chain"],
                   attrs={k: v for k, v in attrs[m].items() if k not in ("experimental_rbf_acceptance", "_shape_chain")})
        if m.startswith("rbf"):
            res["obs_matrix"] = np.round(obs_matrix(corr, obs_rows), 4).tolist()
        if m.endswith("_sector"):
            npz = common.MCMC / f"02_08_sector_input_correlations/{m}_input_matrices.npz"
            if npz.is_file():
                ref = np.load(npz)["correlation"]
                res["max_abs_diff_vs_02_08_npz"] = float(np.abs(ref - corr).max())
        labels = [f"{r['system']}|{r['observable']}|{float(r['center']):+.3f}" for r in rows]
        np.savez_compressed(a.outdir / f"{stem}__{common.SHORT[m]}__matrix.npz", cov=cov, corr=corr, sigma=std,
                            labels=np.array(labels))
        with (a.outdir / f"{stem}__{common.SHORT[m]}__matrix_cov.csv").open("w") as f:
            f.write("bin," + ",".join(labels) + "\n")
            for lab, row in zip(labels, cov):
                f.write(lab + "," + ",".join(f"{v:.6e}" for v in row) + "\n")
        results[m] = res
        print(m, {s: (v["corr_mean"], v["rank_corr"]) for s, v in per_sys.items()}, flush=True)
    illustration(a.proposal, models, attrs, obs_rows, results, a.outdir, stem)
    common.write_manifest(a.outdir, stem, dict(
        common.provenance(models), proposal=a.proposal, models=models, n_bins=len(rows),
        bin_index=str(emu.BIN_INDEX.relative_to(common.PROJECT)), observables=obs_rows,
        sigma=[dict(system=r["system"], observable=r["observable"], center=float(r["center"]),
                    likelihood_std=float(r["likelihood_std"])) for r in rows],
        acceptance=attrs.get("rbf_global", {}).get("experimental_rbf_acceptance"),
        results=results))


if __name__ == "__main__":
    main()
