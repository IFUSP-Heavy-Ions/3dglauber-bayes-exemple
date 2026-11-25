#!/usr/bin/env python3
"""I1-covariance: information item (not a figure ID) on how the experimental covariance of a proposal was built
(DESIGN_SYSTEM 2.08). The main product is the document recipes/I1-covariance__<proposal>.md; this script makes its
illustration and the manifest with every number the document quotes.

For proposal 3 (covinf, learned PHOBOS dN/deta covariance) the figure shows, left to right:
  (a) the data: ln(y_PHOBOS / y_ref) vs centrality for every cross-detector set with PHOBOS (fatia 2a),
      with the fitted offset + centrality slope (the function that was fitted);
  (b) what entered the matrix, per centrality class: common offset a_c, slope term s*g_c and the
      smallest calibration sigma (all in the log space of the likelihood);
  (c) the recipe applied to one class (0-3%): sigma_i(eta) split into the correlated part (a_c) and the
      uncorrelated rest sqrt(d_i).
The manifest (.json) carries every number used (data rows, fit results, per-class components, correlation
summary), so the companion document (recipes/I1-covariance__<proposal>.md) can be checked against it.

The fit is recomputed here from study_nongaussian_systematics/data (same steps as fatia2a_normalizacao.py,
without rewriting its outputs) and checked against fatia2a/results.json.
Usage: python info_covariance.py --proposal covinf --outdir <...>/08_I1-covariance
"""
import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common
import emu
import style

STUDY = common.PROJECT / "study_nongaussian_systematics"
DATA = STUDY / "data"
RESULTS = STUDY / "fatia2a/results.json"
SYSTEM = "Au-Au-200"


def learned_module():
    spec = importlib.util.spec_from_file_location(
        "experimental_learned_covariance", common.PROJECT / "experimental_learned_covariance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fatia2a_pairs():
    """PHOBOS pairs of fatia 2a: r = ln y_A - ln y_B per identical centrality class (same steps as the study)."""
    rows = list(csv.DictReader((DATA / "midrapidity_dndeta.csv").open()))
    for x in rows:
        x["E"], x["lo"], x["hi"], x["y"] = (float(x["sqrtsNN_GeV"]), float(x["cent_lo"]),
                                            float(x["cent_hi"]), float(x["dNdeta"]))
        s = float(x["sys_err_plus"] or 0)
        s2 = float(x["sys_err_minus"] or s)
        sig = np.hypot(0.5 * (s + s2), float(x["stat_err"] or 0))
        x["sig"] = sig / 1.645 if x["sys_cl"] == "90" else sig
    sp = list(csv.DictReader((DATA / "sphenix_dndeta_vs_eta.csv").open()))
    corr = {}
    for k, grp in itertools.groupby(sorted(sp, key=lambda r: (float(r["cent_lo"]), float(r["eta_lo"]))),
                                    key=lambda r: (float(r["cent_lo"]), float(r["cent_hi"]))):
        g = list(grp)
        lo = np.array([float(r["eta_lo"]) for r in g]); hi = np.array([float(r["eta_hi"]) for r in g])
        y = np.array([float(r["dNdeta"]) for r in g])
        w1 = np.clip(np.minimum(hi, 1.0) - np.maximum(lo, -1.0), 0, None)
        w03 = np.clip(np.minimum(hi, 0.3) - np.maximum(lo, -0.3), 0, None)
        corr[k] = (w1 @ y / w1.sum()) / (w03 @ y / w03.sum())
    cv = np.array(list(corr.values()))
    for x in rows:
        if x["experiment"] == "PHOBOS":
            au200 = x["system"] == "Au+Au" and x["E"] == 200
            c = corr.get((x["lo"], x["hi"]), cv.mean()) if au200 else cv.mean()
            x["y_raw"], x["y"] = x["y"], x["y"] / c
            dc = cv.std() if au200 else max(2 * cv.std(), (cv.mean() - 1) / 2)
            x["sig"] = np.hypot(x["sig"], x["y"] * dc)
    by = {}
    for x in rows:
        by.setdefault((x["system"], x["E"], x["lo"], x["hi"]), {})[x["experiment"]] = x
    pairs = {}
    for k, d in by.items():
        if "PHOBOS" not in d:
            continue
        for other in sorted(set(d) - {"PHOBOS"}):
            a, b = d["PHOBOS"], d[other]
            pairs.setdefault((other, k[0], k[1]), []).append(dict(
                lo=k[2], hi=k[3], c=0.5 * (k[2] + k[3]), r=float(np.log(a["y"]) - np.log(b["y"])),
                sr=float(np.hypot(a["sig"] / a["y"], b["sig"] / b["y"])),
                phobos=dict(dNdeta=float(a["y_raw"]), sys90=float(a["sys_err_plus"]), y_eta03=float(a["y"]),
                            sigma1=float(a["sig"]), table=a["source_table"]),
                ref=dict(dNdeta=float(b["y"]), err=float(b["sig"]), table=b["source_table"])))
    out = []
    for (other, sysn, E), L in sorted(pairs.items()):
        L.sort(key=lambda z: z["c"])
        c = np.array([z["c"] for z in L]); r = np.array([z["r"] for z in L]); sr = np.array([z["sr"] for z in L])
        w = 1 / sr**2
        rec = dict(ref=other, system=sysn, E=E, n=len(L), points=L, offset=float(w @ r / w.sum()))
        if len(L) >= 3:
            X = np.vstack([np.ones_like(c), (c - c.mean()) / 100]).T
            beta, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], r * np.sqrt(w), rcond=None)
            res = r - X @ beta
            f = np.sqrt(np.mean(res**2)) / sr.mean()
            rec.update(intercept=float(beta[0]), slope_per_100pct=float(beta[1]), c_mean=float(c.mean()),
                       resid_rms=float(np.sqrt(np.mean(res**2))), rho_eff=float(max(0.0, 1 - f**2)))
        out.append(rec)
    return out


def check_against_study(pairs):
    """Sign convention here is PHOBOS - ref; the study uses alphabetical A - B."""
    ref = {(d["A"], d["B"], d["system"], d["E"]): d for d in json.loads(RESULTS.read_text())["datasets"]}
    for p in pairs:
        key = ("PHENIX", "PHOBOS", p["system"], p["E"]) if p["ref"] < "PHOBOS" else ("PHOBOS", p["ref"], p["system"], p["E"])
        if key not in ref:
            continue
        sign = -1 if p["ref"] < "PHOBOS" else 1
        d = ref[key]
        assert abs(sign * p["offset"] - d["offset"]) < 1e-9, (key, p["offset"], d["offset"])
        if "slope_per_100pct" in d:
            assert abs(sign * p["slope_per_100pct"] - d["slope_per_100pct"]) < 1e-9, key
            assert abs(p["rho_eff"] - d["rho_eff"]) < 1e-9, key


def components(u, slope):
    """Per-class pieces of the learned block, with the likelihood sigmas of bin_index.csv."""
    mod = learned_module()
    rows = [r for r in emu.bins() if r["system"] == SYSTEM]
    groups = []
    for i, r in enumerate(rows):
        if not groups or groups[-1][0] != r["observable"]:
            groups.append([r["observable"], i, i + 1])
        else:
            groups[-1][2] = i + 1
    slices = [(o, slice(a, b)) for o, a, b in groups]
    std = np.array([float(r["likelihood_std"]) for r in rows])
    eta = np.array([float(r["center"]) for r in rows])
    cov = mod.learned_phobos_covariance(SYSTEM, slices, std, u=u, slope=slope)
    blocks = mod._phobos_blocks(slices)
    cmean = np.mean([c for _, _, c in blocks])
    per_class, i0s = [], []
    for obs, slc, c in blocks:
        g = (c - cmean) / 100
        a = float(np.sqrt(std[slc].min() ** 2 - u**2 - (slope * g) ** 2))
        i0 = slc.start + int(np.argmin(np.abs(eta[slc])))
        i0s.append(i0)
        d = std[slc] ** 2 - a**2 - (slope * g) ** 2
        per_class.append(dict(observable=obs, centrality=c, g=g, a=a, slope_term=slope * g,
                              sigma_min=float(std[slc].min()), sigma_eta0=float(std[i0]),
                              sigma_max=float(std[slc].max()), sqrt_d_eta0=float(np.sqrt(d[i0 - slc.start])),
                              sqrt_d_max=float(np.sqrt(d.max())),
                              eta=eta[slc].tolist(), sigma=std[slc].tolist(), sqrt_d=np.sqrt(d).tolist()))
    idx = np.concatenate([np.arange(s.start, s.stop) for _, s, _ in blocks])
    sub = cov[np.ix_(idx, idx)]
    R = sub / np.sqrt(np.outer(np.diag(sub), np.diag(sub)))
    off = R[~np.eye(len(idx), dtype=bool)]
    R0 = cov[np.ix_(i0s, i0s)] / np.outer(std[i0s], std[i0s])
    off0 = R0[~np.eye(len(i0s), dtype=bool)]
    ev = np.linalg.eigvalsh(sub)[::-1]
    a_all = np.array([p["a"] for p in per_class])
    summary = dict(n_bins=int(len(idx)), n_classes=len(blocks), corr_offdiag_mean=float(off.mean()),
                   corr_offdiag_min=float(off.min()), corr_offdiag_max=float(off.max()),
                   corr_eta0_across_classes_min=float(off0.min()), corr_eta0_across_classes_mean=float(off0.mean()),
                   a_min=float(a_all.min()), a_max=float(a_all.max()), a_mean=float(a_all.mean()),
                   largest_eigenvalue_fraction=float(ev[0] / ev.sum()),
                   second_eigenvalue_fraction=float(ev[1] / ev.sum()),
                   diag_unchanged=bool(np.allclose(np.diag(cov), std**2)),
                   corr_eta0_across_classes=np.round(R0, 4).tolist(),
                   eigenvalues_top5=ev[:5].tolist(), trace=float(ev.sum()))
    labels = [f"{rows[i]['observable']}|{float(rows[i]['center']):+.2f}" for i in idx]
    return per_class, summary, dict(cov=sub, corr=R, labels=labels, sigma=std[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", choices=["covinf"], default="covinf")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    style.apply()
    mod = learned_module()
    u, slope = mod.DEFAULTS["u"], mod.DEFAULTS["slope"]
    pairs = fatia2a_pairs()
    check_against_study(pairs)
    per_class, summary, matrix = components(u, slope)
    tch = json.loads(RESULTS.read_text())["three_cornered_hat_AuAu200"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    # (a) data + fitted function
    ax = axes[0]
    shades = plt.get_cmap("Greys")(np.linspace(0.45, 1.0, len(pairs)))
    markers = ["o", "s", "^", "v", "D", "P", "X", "h"]
    handles = []
    for p, col, mk in zip(pairs, shades, markers):
        c = np.array([z["c"] for z in p["points"]]); r = 100 * np.array([z["r"] for z in p["points"]])
        sr = 100 * np.array([z["sr"] for z in p["points"]])
        ax.errorbar(c, r, sr, fmt=mk, ms=4.5, color=col, elinewidth=0.6, capsize=0, alpha=0.9)
        if "slope_per_100pct" in p:
            cc = np.linspace(c.min(), c.max(), 50)
            ax.plot(cc, 100 * (p["intercept"] + p["slope_per_100pct"] * (cc - p["c_mean"]) / 100), color=col, lw=1.6)
        lab = f"PHOBOS/{p['ref']} {p['system']} {p['E']:g}"
        if "rho_eff" in p:
            lab += rf"  ($\rho_{{ef}}$={p['rho_eff']:.2f})"
        handles.append(Line2D([], [], color=col, marker=mk, ls="-", label=lab))
    ax.axhline(0, color="#777777", lw=0.6)
    ax.set_xlabel("centrality (%)")
    ax.set_ylabel(r"$\ln(y_{\rm PHOBOS}/y_{\rm ref})\ \times 100$")
    ax.set_title("(a) data: ratios at $\\eta\\approx0$ + fit offset + slope")
    ax.legend(handles=handles, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    # (b) per-class components
    ax = axes[1]
    c = np.array([p["centrality"] for p in per_class])
    ax.plot(c, [100 * p["sigma_eta0"] for p in per_class], "o-", color="black", lw=1.2, ms=4,
            label=r"$\sigma_i$ calibration at $\eta\approx0$")
    ax.plot(c, [100 * p["a"] for p in per_class], "o-", color=style.LEARNED, lw=2.5, ms=5,
            label=r"$a_c$ common offset ($\rho=1$)")
    ax.plot(c, [100 * abs(p["slope_term"]) for p in per_class], "s--", color=style.LEARNED, lw=1.6, ms=4,
            label=r"$|s\,g_c|$ centrality slope")
    ax.plot(c, [100 * p["sqrt_d_eta0"] for p in per_class], "^:", color="#777777", lw=1.4, ms=4,
            label=r"$\sqrt{d_i}$ uncorrelated at $\eta\approx0$")
    ax.axhline(100 * u, color="#777777", lw=0.8, ls=(0, (4, 3)))
    ax.text(c.max(), 100 * u + 0.15, f"u = {100*u:.2f}% (3-cornered hat)", ha="right", fontsize=9, color="#555555")
    ax.set_ylim(0, 8)
    ax.set_xlabel("centrality class center (%)")
    ax.set_ylabel(r"relative uncertainty (%) = $\sigma$ in $\ln y$")
    ax.set_title("(b) what entered the matrix, per class")
    ax.legend(fontsize=9, loc="center right", frameon=True, framealpha=0.9)
    # (c) recipe on one class
    ax = axes[2]
    p = per_class[0]
    eta = np.array(p["eta"])
    ax.plot(eta, 100 * np.array(p["sigma"]), color="black", lw=1.4, label=r"$\sigma_i$ calibration")
    ax.fill_between(eta, 0, 100 * p["a"], color=style.LEARNED, alpha=0.35, lw=0,
                    label=rf"correlated: $a_c$ = {100*p['a']:.2f}% (flat in $\eta$)")
    ax.plot(eta, 100 * np.array(p["sqrt_d"]), color="#777777", lw=1.4, ls=":", label=r"uncorrelated $\sqrt{d_i}$")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(0, 22)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("relative uncertainty (%)")
    ax.set_title("(c) recipe on one class: Au-Au 0-3% PHOBOS")
    ax.legend(fontsize=9, loc="upper center", frameon=True, framealpha=0.9)
    fig.tight_layout()
    stem = f"I1-covariance__{a.proposal}__learnphob"
    style.save(fig, a.outdir, stem)
    a.outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.outdir / f"{stem}__matrix.npz", **matrix)
    with (a.outdir / f"{stem}__matrix_cov.csv").open("w") as f:   # full numerical result, log space
        f.write("bin," + ",".join(matrix["labels"]) + "\n")
        for lab, row in zip(matrix["labels"], matrix["cov"]):
            f.write(lab + "," + ",".join(f"{v:.6e}" for v in row) + "\n")
    common.write_manifest(a.outdir, stem, dict(
        common.provenance(["learned_phobos"]),
        inputs=dict(cross_detector_table=str((DATA / "midrapidity_dndeta.csv").relative_to(common.PROJECT)),
                    eta_shape=str((DATA / "sphenix_dndeta_vs_eta.csv").relative_to(common.PROJECT)),
                    study_results=str(RESULTS.relative_to(common.PROJECT)),
                    calibration_bins=str(emu.BIN_INDEX.relative_to(common.PROJECT)),
                    covariance_code="experimental_learned_covariance.py"),
        parameters=dict(u=u, slope_per_100pct=slope), three_cornered_hat_AuAu200=tch,
        fit_function="ln(y_PHOBOS/y_ref)(c) = alpha + beta (c - <c>)/100, weights 1/sigma_r^2",
        pairs=pairs, per_class=per_class, correlation_summary=summary,
        matrix_files=[f"{stem}__matrix.npz", f"{stem}__matrix_cov.csv"]))


if __name__ == "__main__":
    main()
