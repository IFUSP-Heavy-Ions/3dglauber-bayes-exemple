"""Variante por detector das Etapas 1-2: normalizacoes independentes para TPC e FTPC.

Motivacao: TPC (|eta|<1) e FTPC (2.7<|eta|<3.9) sao detectores diferentes (calibracao,
eficiencia, tracking, nao-fluxo independentes). Na versao original (sources.py) a fonte
v1 era uma normalizacao unica para os 21 bins, o que acopla as sistematicas dos dois
detectores. Aqui:

    v1a  normalizacao TPC        5% de mu,  so |eta|<1
    v1b  normalizacao FTPC       5% de mu,  so |eta|>2 (unica para East/West)
    v2   assimetria FTPC         igual a sources.py
    v3   borda FTPC              igual a sources.py

A variancia sistematica em cada bin e IDENTICA a da versao original (mesmos 5% de mu em
todos os bins); so muda a estrutura de correlacao entre detectores. Logo sigma_stat
verdadeiro tambem e o mesmo e a comparacao com a cadeia anterior isola o efeito do
desacoplamento.

Uso (da raiz 3dglauber-bayes-example):
    python study_nongaussian_systematics/toy_model/sources_detector_split.py --replicas 5000

Saidas (nao tocam nos arquivos da versao original):
    study_nongaussian_systematics/data/detector_split/{true_sources,replicas,inferred_sources}.npz, .json
    study_nongaussian_systematics/poster/figures/detector_split/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import (EDGE_CENTER, EDGE_WIDTH, OBS, REL_NORM, S_ASYM, S_EDGE, STUDY,  # noqa: E402
                     estimate_pca, estimate_tensor, gamma_shape, load_star, match, sample_lambda)

DATA = STUDY / "data" / "detector_split"
FIG = STUDY / "poster" / "figures" / "detector_split"
NAMES = ["v1a: normalizacao TPC", "v1b: normalizacao FTPC (E+W)", "v2: assimetria FTPC frente/tras",
         "v3: borda da FTPC"]
LABELS = ["lambda_norm_TPC", "lambda_norm_FTPC", "lambda_asym", "lambda_edge"]
SKEW = np.array([2.0, 2.0, 1.5, 2.5])     # mesmas assimetrias da versao original (norm -> as duas norm)


def true_sources(eta, mu, sig):
    fw = np.abs(eta) > 2
    raw = np.stack([
        REL_NORM * mu * ~fw,                                                    # normalizacao TPC
        REL_NORM * mu * fw,                                                     # normalizacao FTPC
        S_ASYM * sig * np.sign(eta) * fw,                                       # +frente / -tras
        S_EDGE * sig * np.exp(-((np.abs(eta) - EDGE_CENTER) / EDGE_WIDTH) ** 2) * fw,  # borda
    ], axis=1)
    s = np.linalg.norm(raw, axis=0)
    return raw / s, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rank = len(NAMES)

    eta, mu, sig_tot, sig_orig = load_star()
    V, s = true_sources(eta, mu, sig_tot)
    sys_var = (V**2 * s**2).sum(1)
    share = sys_var / sig_tot**2
    assert share.max() < 0.9, f"fontes excedem 90% da variancia informada: {share.max():.2f}"
    sig_stat = np.sqrt(sig_tot**2 - sys_var)
    np.savez(DATA / "true_sources.npz", eta=eta, v_hat=V, s=s, skew=SKEW, k=gamma_shape(SKEW),
             sigma_total=sig_tot, sigma_stat=sig_stat, names=NAMES, labels=LABELS)

    lam = sample_lambda(rng, s, SKEW, args.replicas)
    reps = lam @ V.T + rng.normal(0, sig_stat, size=(args.replicas, len(eta)))
    np.savez(DATA / "replicas.npz", replicas=reps, lam=lam, seed=args.seed)

    Vt, st, gt = estimate_tensor(reps, rank, rng)
    perm, ang_t = match(V, Vt)
    Vt, st, gt = Vt[:, perm], st[perm], gt[perm]
    Vt = Vt * np.sign(np.sum(Vt * V, axis=0))
    Vp, sp, gp = estimate_pca(reps, rank)
    perm_p, ang_p = match(V, Vp)

    Ns = [300, 1000, 3000, 10000]
    curve = {"N": Ns, "tensor_deg": [], "pca_deg": []}
    for N in Ns:
        at, ap_ = [], []
        for rep in range(5):
            rr = np.random.default_rng(10_000 * N + rep)
            r = sample_lambda(rr, s, SKEW, N) @ V.T + rr.normal(0, sig_stat, size=(N, len(eta)))
            at.append(match(V, estimate_tensor(r, rank, rr)[0])[1].mean())
            ap_.append(match(V, estimate_pca(r, rank)[0])[1].mean())
        curve["tensor_deg"].append([float(np.mean(at)), float(np.std(at))])
        curve["pca_deg"].append([float(np.mean(ap_)), float(np.std(ap_))])

    sys_inf = (Vt**2 * st**2).sum(1)
    sig_stat_inf = np.sqrt(np.maximum(sig_tot**2 - sys_inf, (0.05 * sig_tot) ** 2))
    np.savez(DATA / "inferred_sources.npz", eta=eta, v_hat=Vt, s=st, skew=gt, k=gamma_shape(gt),
             sigma_total=sig_tot, sigma_stat=sig_stat_inf, observable=OBS, n_replicas=args.replicas,
             labels=np.array(LABELS), names=np.array(NAMES))
    report = dict(
        variant="detector_split", observable=OBS, n_bins=len(eta), n_replicas=args.replicas, seed=args.seed,
        variance_share_systematic=dict(min=float(share.min()), max=float(share.max())),
        sources=[dict(name=NAMES[a], label=LABELS[a], s_true=float(s[a]), s_tensor=float(st[a]),
                      skew_true=float(SKEW[a]), skew_tensor=float(gt[a]), k_tensor=float(gamma_shape(gt[a])),
                      angle_tensor_deg=float(ang_t[a]), angle_pca_deg=float(ang_p[a])) for a in range(rank)],
        recovery_vs_N=curve,
    )
    (DATA / "inferred_sources.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(1, rank, figsize=(4.6 * rank, 3.8))
    for a in range(rank):
        ax[a].bar(eta, s[a] * V[:, a] * 1e3, width=0.17, color="#bbb", label="verdadeira")
        ax[a].plot(eta, st[a] * Vt[:, a] * 1e3, "o", color="#2e86ab", label="Tensor PCA (p=3)")
        ax[a].plot(eta, sp[perm_p[a]] * Vp[:, perm_p[a]] * np.sign(Vp[:, perm_p[a]] @ V[:, a]) * 1e3, "x",
                   color="#edae49", label="PCA (p=2)")
        ax[a].set_title(f"{NAMES[a]}\nerro: tensor {ang_t[a]:.1f}°, PCA {ang_p[a]:.1f}°", fontsize=10)
        ax[a].set_xlabel(r"$\eta$")
    ax[0].set_ylabel(r"$s_a\,\hat v_a$  ($\times10^{-3}$ em $v_2$)")
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "sources_true_vs_recovered.png", dpi=160)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for key, c, lab in (("tensor_deg", "#2e86ab", r"Tensor PCA ($\hat\kappa_3$)"), ("pca_deg", "#edae49", "PCA (covariancia)")):
        m = np.array(curve[key]); ax.errorbar(Ns, m[:, 0], yerr=m[:, 1], marker="o", color=c, label=lab, capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("replicas N"); ax.set_ylabel("erro medio de direcao (graus)")
    ax.set_title("STAR v2(eta), fontes por detector"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "recovery_vs_N.png", dpi=160)

    print(json.dumps({k: report[k] for k in ("variance_share_systematic", "sources")}, indent=1))
    print(json.dumps(curve))


if __name__ == "__main__":
    main()
