"""Fatia 2a: normalizacao de dN/deta em eta~0 a partir de comparacoes entre detectores.

r = ln y_A - ln y_B  (mesmo sistema, energia e classe de centralidade)
- PHOBOS: sys 90% C.L. -> /1.645 ; janela |eta|<1 -> corrigida para |eta|<0.3 com a forma do sPHENIX (Au+Au 200)
- Por conjunto (par x sistema x energia): offset a (ponderado), inclinacao b em centralidade,
  dispersao residual -> fracao nao correlacionada entre classes -> rho efetivo (comparar com CS rho=1)
- Three-cornered hat em Au+Au 200 (PHOBOS, PHENIX, sPHENIX)
- 3o momento: assimetria dos residuos (apenas relativa; ver MAPA.tex)
"""
import csv, json, itertools, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = HERE
rng = np.random.default_rng(1)

rows = list(csv.DictReader(open(os.path.join(DATA, "midrapidity_dndeta.csv"))))
for x in rows:
    x["E"] = float(x["sqrtsNN_GeV"]); x["lo"] = float(x["cent_lo"]); x["hi"] = float(x["cent_hi"])
    x["y"] = float(x["dNdeta"])
    s = float(x["sys_err_plus"] or 0); s2 = float(x["sys_err_minus"] or s)
    st = float(x["stat_err"] or 0)
    sig = np.sqrt((0.5 * (s + s2)) ** 2 + st ** 2)
    if x["sys_cl"] == "90": sig /= 1.645
    x["sig"] = sig

# ---- correcao de janela em eta (forma do sPHENIX) ----
sp = list(csv.DictReader(open(os.path.join(DATA, "sphenix_dndeta_vs_eta.csv"))))
corr = {}
for key, grp in itertools.groupby(sorted(sp, key=lambda r: (float(r["cent_lo"]), float(r["eta_lo"]))),
                                  key=lambda r: (float(r["cent_lo"]), float(r["cent_hi"]))):
    g = list(grp)
    eta_c = np.array([(float(r["eta_lo"]) + float(r["eta_hi"])) / 2 for r in g])
    y = np.array([float(r["dNdeta"]) for r in g])
    w1 = np.clip(np.minimum(np.array([float(r["eta_hi"]) for r in g]), 1.0) -
                 np.maximum(np.array([float(r["eta_lo"]) for r in g]), -1.0), 0, None)
    w03 = np.clip(np.minimum(np.array([float(r["eta_hi"]) for r in g]), 0.3) -
                  np.maximum(np.array([float(r["eta_lo"]) for r in g]), -0.3), 0, None)
    corr[key] = (np.sum(w1 * y) / w1.sum()) / (np.sum(w03 * y) / w03.sum())
cvals = np.array(list(corr.values()))
c_mean, c_sd = cvals.mean(), cvals.std()
for x in rows:
    if x["experiment"] == "PHOBOS":
        k = (x["lo"], x["hi"])
        c = corr.get(k, c_mean) if (x["system"] == "Au+Au" and x["E"] == 200) else c_mean
        x["y_raw"] = x["y"]; x["y"] = x["y"] / c
        # incerteza da correcao: espalhamento entre classes; dobrada fora de Au+Au 200
        dc = c_sd if (x["system"] == "Au+Au" and x["E"] == 200) else max(2 * c_sd, (c_mean - 1) / 2)
        x["sig"] = np.hypot(x["sig"], x["y"] * dc)

# ---- pares com classe identica ----
def key(x): return (x["system"], x["E"], x["lo"], x["hi"])
by = {}
for x in rows: by.setdefault(key(x), {})[x["experiment"]] = x
pairs = {}
for k, d in by.items():
    for A, B in itertools.combinations(sorted(d), 2):
        a, b = d[A], d[B]
        r = np.log(a["y"]) - np.log(b["y"])
        sr = np.hypot(a["sig"] / a["y"], b["sig"] / b["y"])
        pairs.setdefault((A, B, k[0], k[1]), []).append(dict(c=0.5 * (k[2] + k[3]), lo=k[2], hi=k[3], r=r, sr=sr,
                                                               sA=a["sig"] / a["y"], sB=b["sig"] / b["y"]))

summary = []
for (A, B, sysn, E), L in sorted(pairs.items()):
    L.sort(key=lambda z: z["c"])
    c = np.array([z["c"] for z in L]); r = np.array([z["r"] for z in L]); sr = np.array([z["sr"] for z in L])
    w = 1 / sr ** 2
    a = np.sum(w * r) / w.sum()
    rec = dict(A=A, B=B, system=sysn, E=E, n=len(L), offset=a, offset_pct=100 * (np.exp(a) - 1),
               sigma_quoted_mean=float(sr.mean()), pull_offset=a / sr.mean(),
               classes=[(z["lo"], z["hi"]) for z in L], r=r.tolist(), sr=sr.tolist(), c=c.tolist())
    if len(L) >= 3:
        X = np.vstack([np.ones_like(c), (c - c.mean()) / 100]).T
        beta, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], r * np.sqrt(w), rcond=None)
        res = r - X @ beta
        rec.update(slope_per_100pct=float(beta[1]), resid_rms=float(np.sqrt(np.mean(res ** 2))),
                   resid=res.tolist())
        # fracao nao correlacionada: rms residual / sigma citado ; rho_eff = 1 - f^2
        f = rec["resid_rms"] / rec["sigma_quoted_mean"]
        rec.update(f_uncorr=float(f), rho_eff=float(max(0.0, 1 - f ** 2)))
    summary.append(rec)

# ---- three-cornered hat, Au+Au 200 ----
tch = {}
trio = ["PHENIX", "PHOBOS", "sPHENIX"]
common = [k for k, d in by.items() if k[0] == "Au+Au" and k[1] == 200 and all(t in d for t in trio)]
common.sort(key=lambda k: k[2])
lr = {t: np.array([np.log(by[k][t]["y"]) for k in common]) for t in trio}
def var_detrended(v):
    c = np.array([0.5 * (k[2] + k[3]) for k in common]); X = np.vstack([np.ones_like(c), c]).T
    beta, *_ = np.linalg.lstsq(X, v, rcond=None); res = v - X @ beta
    return res.var(ddof=2), res
V = {}; RES = {}
for A, B in itertools.combinations(trio, 2):
    V[(A, B)], RES[(A, B)] = var_detrended(lr[A] - lr[B])
vA = 0.5 * (V[("PHENIX", "PHOBOS")] + V[("PHENIX", "sPHENIX")] - V[("PHOBOS", "sPHENIX")])
vB = 0.5 * (V[("PHENIX", "PHOBOS")] + V[("PHOBOS", "sPHENIX")] - V[("PHENIX", "sPHENIX")])
vC = 0.5 * (V[("PHENIX", "sPHENIX")] + V[("PHOBOS", "sPHENIX")] - V[("PHENIX", "PHOBOS")])
quoted = {t: float(np.mean([by[k][t]["sig"] / by[k][t]["y"] for k in common])) for t in trio}
offs = {t: None for t in trio}
tch = dict(classes=[(k[2], k[3]) for k in common],
           var_class_to_class={"PHENIX": vA, "PHOBOS": vB, "sPHENIX": vC},
           sd_class_to_class_pct={t: 100 * float(np.sqrt(max(v, 0))) for t, v in zip(trio, [vA, vB, vC])},
           quoted_rel_err_pct={t: 100 * quoted[t] for t in trio},
           pair_offsets_pct={f"{A}-{B}": 100 * float(np.exp(np.mean(lr[A] - lr[B])) - 1) for A, B in itertools.combinations(trio, 2)})

# ---- 3o momento (relativo): assimetria dos residuos padronizados ----
allres = np.concatenate([np.array(s["resid"]) / np.array(s["sr"]) for s in summary if "resid" in s])
def skew(v): v = v - v.mean(); return float(np.mean(v ** 3) / np.mean(v ** 2) ** 1.5)
bs = [skew(rng.choice(allres, len(allres))) for _ in range(2000)]
offs_all = np.array([s["pull_offset"] for s in summary if s["n"] >= 3])
third = dict(n_resid=int(len(allres)), skew_resid=skew(allres), skew_resid_bootstrap_sd=float(np.std(bs)),
             n_offsets=int(len(offs_all)), skew_offsets=skew(offs_all) if len(offs_all) > 2 else None)

out = dict(eta_window_correction=dict(mean=float(c_mean), sd=float(c_sd), per_class={f"{k[0]:g}-{k[1]:g}": float(v) for k, v in corr.items()}),
           datasets=[{k: v for k, v in s.items() if k not in ("r", "sr", "c", "resid")} for s in summary],
           three_cornered_hat_AuAu200=tch, third_moment=third)
json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)

# ---- figura ----
big = [s for s in summary if s["n"] >= 3]
ncol = 4; nrow = int(np.ceil(len(big) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.7 * nrow), sharey=True, squeeze=False)
for ax, s in zip(axs.flat, big):
    c = np.array(s["c"]); r = 100 * np.array(s["r"]); sr = 100 * np.array(s["sr"])
    ax.fill_between(c, -sr, sr, color="0.85", label="±σ citado (soma)")
    ax.errorbar(c, r, fmt="o", ms=3, color="C0")
    ax.axhline(100 * s["offset"], color="C3", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{s['A']}−{s['B']}  {s['system']} {s['E']:g}", fontsize=8)
    ax.text(0.03, 0.04, f"offset {s['offset_pct']:+.1f}%  ρ_ef {s.get('rho_eff', float('nan')):.2f}", transform=ax.transAxes, fontsize=7)
for ax in axs.flat[len(big):]: ax.axis("off")
for ax in axs[-1]: ax.set_xlabel("centralidade (%)")
for ax in axs[:, 0]: ax.set_ylabel("ln(y_A/y_B) ×100")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "razoes_por_conjunto.pdf")); fig.savefig(os.path.join(OUT, "razoes_por_conjunto.png"), dpi=130)
print(json.dumps(out, indent=1, default=float)[:6000])
