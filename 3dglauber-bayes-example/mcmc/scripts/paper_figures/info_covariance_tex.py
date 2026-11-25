#!/usr/bin/env python3
"""Tables and numeric macros for the I1-covariance article (DESIGN_SYSTEM 2.08).

Reads the manifest written by info_covariance.py and writes LaTeX snippets into <recipe>/tab/, so the article
(recipes/I1-covariance__<proposal>/I1-covariance__<proposal>.tex) never carries a hand-copied number.
Runs locally (only json + stdlib).
Usage: python info_covariance_tex.py --manifest <...>/I1-covariance__covinf__learnphob.json \
           --recipe recipes/I1-covariance__covinf
"""
import argparse
import json
from pathlib import Path

CLASSES = ["00--03", "03--06", "06--10", "10--15", "15--20", "20--25", "25--30", "30--35", "35--40", "40--45", "45--50"]


def num(x, n=2, sign=False):
    s = f"{x:+.{n}f}" if sign else f"{x:.{n}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def pct(x, n=2, sign=False):
    return num(100 * x, n, sign)


def macros(d):
    S, T, P = d["correlation_summary"], d["three_cornered_hat_AuAu200"], d["parameters"]
    pairs = d["pairs"]
    n_cmp = sum(p["n"] for p in pairs)
    phobos_rows = {(p["system"], p["E"], z["lo"], z["hi"]) for p in pairs for z in p["points"]}
    rho = [p["rho_eff"] for p in pairs if "rho_eff" in p]
    res = [p["resid_rms"] for p in pairs if "resid_rms" in p]
    s200 = {p["ref"]: p["slope_per_100pct"] for p in pairs if p["E"] == 200 and p["system"] == "Au+Au"}
    m = dict(
        valU=pct(P["u"]), valS=num(P["slope_per_100pct"], 3), valNbins=str(S["n_bins"]), valNclasses=str(S["n_classes"]),
        valNsets=str(len(pairs)), valNcmp=str(n_cmp), valNphobos=str(len(phobos_rows)),
        valRhoMin=num(min(rho)), valRhoMax=num(max(rho)), valResMin=pct(min(res), 1), valResMax=pct(max(res), 1),
        valAmin=pct(S["a_min"]), valAmax=pct(S["a_max"]), valAmean=pct(S["a_mean"]),
        valAsq=f"{S['a_mean']**2:.4f}".replace(".", "{,}"),
        valEigOne=num(100 * S["largest_eigenvalue_fraction"], 1), valEigTwo=num(100 * S["second_eigenvalue_fraction"], 1),
        valCorrMean=num(S["corr_offdiag_mean"]), valCorrMin=num(S["corr_offdiag_min"]), valCorrMax=num(S["corr_offdiag_max"]),
        valCorrEtaMin=num(S["corr_eta0_across_classes_min"]), valCorrEtaMean=num(S["corr_eta0_across_classes_mean"]),
        valSlopePHENIX=pct(s200.get("PHENIX", float("nan")), 1, True), valSlopeSPHENIX=pct(s200.get("sPHENIX", float("nan")), 1, True),
        valTchPHOBOS=num(T["sd_class_to_class_pct"]["PHOBOS"]), valTchPHENIX=num(T["sd_class_to_class_pct"]["PHENIX"]),
        valTchSPHENIX=num(T["sd_class_to_class_pct"]["sPHENIX"]),
        valCmean=num(sum(c["centrality"] for c in d["per_class"]) / len(d["per_class"])),
        valSlopeMax=pct(max(abs(c["slope_term"]) for c in d["per_class"])),
    )
    m = {k: v.replace(".", "{,}") for k, v in m.items()}
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())


def tab_data(d):
    out = []
    for p in d["pairs"]:
        ph, rf = p["points"][0]["phobos"]["table"], p["points"][0]["ref"]["table"].replace("_", "\\_")
        energy = f"{p['E']:g}".replace(".", "{,}")
        out.append(f"\\midrule\n\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{PHOBOS/{p['ref']}, {p['system']} {energy}\\,GeV}} "
                   f"--- {p['n']} classes}}\\\\\n\\multicolumn{{6}}{{@{{}}l}}{{\\footnotesize PHOBOS: arXiv:{ph.replace('1011.1940 ', '1011.1940, ')}; "
                   f"{p['ref']}: {rf}}}\\\\\n\\midrule\n")
        for z in p["points"]:
            out.append((f"{z['lo']:g}--{z['hi']:g} & {z['phobos']['dNdeta']:g} $\\pm$ {z['phobos']['sys90']:g} & "
                        f"{num(z['phobos']['y_eta03'], 1)} & {num(z['phobos']['sigma1'], 1)} & "
                        f"{z['ref']['dNdeta']:g} $\\pm$ {num(z['ref']['err'], 1)} & {pct(z['r'], 2, True)} $\\pm$ {pct(z['sr'])} \\\\\n"
                        ).replace(".", "{,}"))
    return "".join(out)


def tab_fit(d):
    rows = []
    for p in d["pairs"]:
        rows.append(f"PHOBOS/{p['ref']} & {p['system']} & {p['E']:g} & {p['n']} & "
                    f"{pct(p['offset'], 1, True)} & {pct(p['slope_per_100pct'], 1, True)} & {pct(p['resid_rms'])} & {num(p['rho_eff'])} \\\\\n")
    return "".join(rows).replace(".", "{,}")


def tab_classes(d):
    rows = []
    for lab, c in zip(CLASSES, d["per_class"]):
        rows.append(f"{lab} & {num(c['centrality'], 1)} & {num(c['g'], 3, True)} & {pct(c['sigma_eta0'])} & {pct(c['sigma_min'])} & "
                    f"{pct(c['sigma_max'], 1)} & \\textbf{{{pct(c['a'])}}} & {pct(c['slope_term'], 2, True)} & {pct(c['sqrt_d_eta0'])} \\\\\n")
    return "".join(rows).replace(".", "{,}")


def tab_corr11(d):
    R = d["correlation_summary"]["corr_eta0_across_classes"]
    head = " & " + " & ".join(f"\\rotatebox{{90}}{{{c}}}" for c in CLASSES) + " \\\\\n\\midrule\n"
    body = "".join(f"{CLASSES[i]} & " + " & ".join(num(v) for v in row) + " \\\\\n" for i, row in enumerate(R))
    return (head + body).replace(".", "{,}")


def tab_bins(d, key):
    pcs = d["per_class"]
    eta = pcs[0]["eta"]
    assert all(len(c[key]) == len(eta) for c in pcs)
    head = "$\\eta$ & " + " & ".join(CLASSES) + " \\\\\n\\midrule\n\\endhead\n"
    body = "".join(f"{num(e, 2, True)} & " + " & ".join(num(100 * c[key][k], 2) for c in pcs) + " \\\\\n"
                   for k, e in enumerate(eta))
    return (head + body).replace(".", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--recipe", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.manifest.read_text())
    tab = a.recipe / "tab"
    tab.mkdir(parents=True, exist_ok=True)
    files = {"valores.tex": macros(d), "dados.tex": tab_data(d), "ajuste.tex": tab_fit(d),
             "classes.tex": tab_classes(d), "corr11.tex": tab_corr11(d),
             "bins_sigma.tex": tab_bins(d, "sigma"), "bins_sqrtd.tex": tab_bins(d, "sqrt_d")}
    for name, text in files.items():
        (tab / name).write_text("% generated by info_covariance_tex.py from " + a.manifest.name + "; do not edit\n" + text)
        print("wrote", tab / name)


if __name__ == "__main__":
    main()
