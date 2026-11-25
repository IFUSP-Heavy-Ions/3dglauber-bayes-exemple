#!/usr/bin/env python3
"""Tables and numeric macros for the I1-covariance article of proposals 1 (global) and 2 (sector) (DESIGN_SYSTEM 2.08).

Reads the manifest of info_covariance_assumed.py and writes LaTeX snippets into <recipe>/tab/; the article never carries
a hand-copied number. Runs locally (json + stdlib).
Usage: python info_covariance_assumed_tex.py --manifest <...>/I1-covariance__global__cs-rbf.json --recipe recipes/I1-covariance__global
"""
import argparse
import json
import re
from pathlib import Path

SYS = {"Au-Au-200": "Au-Au", "d-Au-200": "d-Au"}
SP = {"pi": r"$\pi^\pm$", "k": r"$K^\pm$", "p": r"$p/\bar p$"}
AXIS = {"pt": r"$p_T$", "eta": r"$\eta$", "centrality": "centr."}
UNIT = {"pt": "GeV/$c$", "eta": "", "centrality": "\\%"}


def c(x, n=2, sign=False):
    s = f"{x:+.{n}f}" if sign else f"{x:.{n}f}"
    return s.replace(".", "{,}").replace("-", "$-$").replace("+", "$+$")


def name(obs):
    cen = re.search(r"cen_(\d+)_(\d+)", obs)
    cent = f" {int(cen.group(1))}--{int(cen.group(2))}\\%" if cen else ""
    if obs.startswith("dNdeta"):
        return r"$dN/d\eta(\eta)$" + cent
    if obs.startswith("v22_eta"):
        return r"$v_2(\eta)$" + cent
    if obs.startswith("v22_pt"):
        return r"$v_2(p_T)$" + cent
    if obs.startswith("v22_int"):
        return r"$v_2$(centr.)"
    if obs.startswith("v32_int"):
        return r"$v_3$(centr.)"
    return SP[obs.split("_")[1]] + r" $\langle p_T\rangle$(centr.)"


def sig_fmt(obs, v):
    """sigma_i in the likelihood space: multiplicities in log -> %, everything else absolute -> x10^-3."""
    return c(100 * v, 1) if obs.startswith("dN") else c(1000 * v, 1)


def sig_unit(obs):
    return r"\%" if obs.startswith("dN") else r"$10^{-3}$"


def macros(d):
    R = d["results"]
    m = dict(valNbins=str(d["n_bins"]), valNobs=str(len(d["observables"])),
             valNau=str(sum(o["n"] for o in d["observables"] if o["system"] == "Au-Au-200")),
             valNda=str(sum(o["n"] for o in d["observables"] if o["system"] == "d-Au-200")))
    for model, r in R.items():
        tag = "Cs" if model.startswith("cs") else "Rbf"
        for sysn, key in (("Au-Au-200", "Au"), ("d-Au-200", "Da")):
            s = r["per_system"][sysn]
            m[f"val{tag}{key}Mean"] = c(s["corr_mean"], 3)
            m[f"val{tag}{key}Half"] = c(100 * s["frac_above_0p5"], 1)
            m[f"val{tag}{key}Zero"] = c(100 * s["frac_exact_zero"], 1)
            m[f"val{tag}{key}Rank"] = str(s["rank_corr"])
            m[f"val{tag}{key}Eig"] = c(100 * s["top_eig_frac_corr"][0], 1)
        a = r["attrs"]
        if "experimental_cs_rho" in a:
            m["valRho"] = c(float(a["experimental_cs_rho"]), 0)
        if "experimental_rbf_lengths" in a:
            L = a["experimental_rbf_lengths"]
            m.update(valLpt=c(L["pt"], 1), valLeta=c(L["eta"], 0), valLcent=c(L["centrality"], 0))
        if "experimental_covariance_signature" in a:
            m[f"valSig{tag}"] = a["experimental_covariance_signature"][:12]
        m[f"valSteps{tag}"] = f"{r['chain_shape'][0]} $\\times$ {r['chain_shape'][1]:,}".replace(",", "\\,")
        if "max_abs_diff_vs_02_08_npz" in r:
            v = r["max_abs_diff_vs_02_08_npz"]
            m[f"valDiff{tag}"] = "0 (idênticas)" if v == 0 else f"{v:.1e}".replace(".", "{,}")
    if any("regions" in o for o in d["observables"]):
        m["valNsectors"] = str(sum(len(o["regions"]["sizes"]) for o in d["observables"]))
        m["valNsplit"] = str(sum(1 for o in d["observables"] if o["regions"]["boundaries"]))
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())


def tab_obs(d):
    """One row per observable: data description + what each model does to it."""
    models = list(d["results"])
    rows = []
    for k, o in enumerate(d["observables"]):
        pr = [d["results"][m]["per_observable"][k] for m in models]
        base = (f"{k} & {SYS[o['system']]} & {name(o['observable'])} & {o['experiment']} & {o['n']} & {AXIS[o['axis']]} & "
                f"{c(o['x_min'], 2 if o['axis'] != 'centrality' else 1)}--{c(o['x_max'], 2 if o['axis'] != 'centrality' else 1)} & "
                f"{sig_fmt(o['observable'], o['sigma_min'])}--{sig_fmt(o['observable'], o['sigma_max'])} {sig_unit(o['observable'])}")
        if d["proposal"] == "global":
            cs, rbf = pr
            extra = (f" & {c(rbf['neighbour_mean'], 2) if o['n'] > 1 else '---'} & {c(rbf['within_mean'], 2) if o['n'] > 1 else '---'}"
                     f" & {c(rbf['cross_mean'], 3)}")
        else:
            cs, rbf = pr
            reg = o["regions"]
            seg = "+".join(str(n) for _, n in reg["sizes"])
            extra = (f" & {seg} & {c(cs['within_mean'], 2) if o['n'] > 1 else '---'}"
                     f" & {c(rbf['neighbour_mean'], 2) if o['n'] > 1 else '---'} & {c(rbf['within_mean'], 2) if o['n'] > 1 else '---'}")
        rows.append(base + extra + " \\\\\n")
    return "".join(rows)


def tab_coords(d):
    """Global RBF: the coordinates of every observable in (pT, eta, centrality) and the acceptance window used."""
    acc = d["acceptance"]
    rows = []
    for k, o in enumerate(d["observables"]):
        def coord(ax):
            if o["axis"] == ax:
                return "bin"
            v = o[f"coord_{'cent' if ax == 'centrality' else ax}"]
            return ", ".join(c(x, 3 if ax == "pt" else 2) for x in v)
        win = acc[o["system"]].get(o["observable"], {})
        src = []
        for ax in ("pt", "eta"):
            if ax in win:
                lo, hi = win[ax]["window"]
                src.append(f"{AXIS[ax]} $\\in$ [{c(lo, 2)}, {c(hi, 2)}]")
        rows.append(f"{k} & {SYS[o['system']]} & {name(o['observable'])} & {o['experiment']} & {coord('pt')} & {coord('eta')} & "
                    f"{coord('centrality')} & {'; '.join(src) or '---'} \\\\\n")
    return "".join(rows)


def tab_refs(d):
    """Global RBF: the reference behind every acceptance window (one row per distinct reference)."""
    seen = {}
    for sysn, obs_map in d["acceptance"].items():
        for obs, axes in obs_map.items():
            for ax, w in axes.items():
                key = (w.get("ref") or w.get("source"), w.get("source"))
                seen.setdefault(key, []).append(f"{SYS[sysn]} {name(obs)}")
    rows = []
    for (ref, source), obs in seen.items():
        txt = (ref or "").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("<pT>", r"$\langle p_T\rangle$")
        s = (source or "").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
        rows.append(f"{txt} & {'(janela da própria publicação)' if s == txt else s} & {len(obs)} \\\\\n")
    return "".join(rows)


def tab_regions(d):
    """Sector: the observables split into rapidity regions (boundaries, labels, sizes)."""
    rows = []
    for k, o in enumerate(d["observables"]):
        reg = o["regions"]
        if not reg["boundaries"]:
            continue
        b = ", ".join(c(x, 0) for x in reg["boundaries"])
        layout = " | ".join(f"{lab} ({n})" for lab, n in reg["layout"])
        sizes = ", ".join(f"{lab} {n}" for lab, n in reg["sizes"])
        rows.append(f"{k} & {SYS[o['system']]} & {name(o['observable'])} & {o['experiment']} & {o['n']} & [{b}] & {layout} & {sizes} \\\\\n")
    return "".join(rows)


def tab_systems(d):
    rows = []
    for model, r in d["results"].items():
        lab = {"cs_global": "CS global", "rbf_global": "RBF global", "cs_sector": "CS sector", "rbf_sector": "RBF sector"}[model]
        for sysn, s in r["per_system"].items():
            rows.append(f"{lab} & {SYS[sysn]} & {s['n_bins']} & {c(s['corr_mean'], 3)} & {c(100 * s['frac_above_0p5'], 1)} & "
                        f"{c(100 * s['frac_exact_zero'], 1)} & {s['rank_corr']} & {c(100 * s['top_eig_frac_corr'][0], 1)} \\\\\n")
    return "".join(rows)


def tab_pairs(d, top=15):
    """Global RBF: most correlated pairs of different observables (mean R over the pair of blocks)."""
    M = d["results"]["rbf_global"]["obs_matrix"]
    obs = d["observables"]
    pairs = sorted(((M[i][j], i, j) for i in range(len(obs)) for j in range(i + 1, len(obs))
                    if obs[i]["system"] == obs[j]["system"]), reverse=True)[:top]
    return "".join(f"{SYS[obs[i]['system']]} & {name(obs[i]['observable'])} ({obs[i]['experiment']}) & "
                   f"{name(obs[j]['observable'])} ({obs[j]['experiment']}) & {c(v, 3)} \\\\\n" for v, i, j in pairs)


def tab_sigma(d):
    """Appendix: sigma_i (%, likelihood space) bin by bin, one paragraph per observable."""
    out, k = [], 0
    by = {}
    for s in d["sigma"]:
        by.setdefault((s["system"], s["observable"]), []).append(s)
    for o in d["observables"]:
        vals = by[(o["system"], o["observable"])]
        n = 2 if o["axis"] != "centrality" else 1
        f = (lambda v: c(100 * v, 2)) if o["observable"].startswith("dN") else (lambda v: c(1000 * v, 3))
        body = "; ".join(f"{c(v['center'], n)}: {f(v['likelihood_std'])}" for v in vals)
        out.append(f"\\textbf{{{k}. {SYS[o['system']]} {name(o['observable'])}, {o['experiment']}}} "
                   f"({AXIS[o['axis']]}: $\\sigma_i$ em {sig_unit(o['observable'])}) {body}.\\par\n")
        k += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--recipe", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.manifest.read_text())
    tab = a.recipe / "tab"
    tab.mkdir(parents=True, exist_ok=True)
    files = {"valores.tex": macros(d), "observaveis.tex": tab_obs(d), "sistemas.tex": tab_systems(d),
             "sigma.tex": tab_sigma(d)}
    if d["proposal"] == "sector":
        files["regioes.tex"] = tab_regions(d)
    if d["proposal"] == "global":
        files.update({"coordenadas.tex": tab_coords(d), "referencias.tex": tab_refs(d), "pares.tex": tab_pairs(d)})
    for fname, text in files.items():
        (tab / fname).write_text(f"% generated by info_covariance_assumed_tex.py from {a.manifest.name}; do not edit\n" + text)
        print("wrote", tab / fname)


if __name__ == "__main__":
    main()
