"""Etapas 1-2: fontes sistematicas nao gaussianas no STAR v2(eta), replicas e Tensor PCA.

Uso (da raiz 3dglauber-bayes-example):
    python study_nongaussian_systematics/toy_model/sources.py [--replicas 1000] [--seed 2026]

Saidas em study_nongaussian_systematics/data/:
    star_v2_eta.csv           eta, v2, sigma_total (variancia informada), sigma_original
    true_sources.npz          fontes verdadeiras (direcoes, escalas, assimetrias)
    replicas.npz              replicas sinteticas
    inferred_sources.npz/json fontes inferidas por Tensor PCA (entrada da cadeia com nuisance)
e figuras em study_nongaussian_systematics/poster/figures/.

Orcamento de erro: a variancia informada em cada bin e tratada como
sigma_total^2 = sigma_stat^2 + sum_a s_a^2 vhat_ai^2. As fontes geram as replicas; o
estatistico e o que sobra. Na likelihood com nuisance, os lambda_a absorvem a parte
sistematica e a covariancia leva so sigma_stat^2 (sem contar o sistematico duas vezes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parent
DATA = STUDY / "data"
FIG = STUDY / "poster" / "figures"
OBS = "v22_eta_cen_20_70_STAR"
ETA_EDGES = np.array([[-3.9, -3.7], [-3.7, -3.5], [-3.5, -3.3], [-3.3, -3.1], [-3.1, -2.9], [-2.9, -2.7],
                      [-0.9, -0.7], [-0.7, -0.5], [-0.5, -0.3], [-0.3, -0.1], [-0.1, 0.1], [0.1, 0.3],
                      [0.3, 0.5], [0.5, 0.7], [0.7, 0.9],
                      [2.7, 2.9], [2.9, 3.1], [3.1, 3.3], [3.3, 3.5], [3.5, 3.7], [3.7, 3.9]])
NAMES = ["v1: normalizacao global", "v2: assimetria FTPC frente/tras", "v3: borda da FTPC"]

# Escolhas do toy (fixas e documentadas)
REL_NORM = 0.05                      # v1: 5% de normalizacao (relativo a v2)
S_ASYM = 0.55                        # v2: rms em unidades de sigma_total nos bins da FTPC
S_EDGE = 0.60                        # v3: rms em unidades de sigma_total no pico (borda |eta|=3.9)
EDGE_CENTER, EDGE_WIDTH = 3.9, 0.45
SKEW = np.array([2.0, 1.5, 2.5])     # assimetria de cada amplitude lambda_a (Gamma)


# ------------------------------------------------------------------------------- data
def load_star():
    def read(name):
        rows = [l.split() for l in (ROOT / "HIC_experimental_data/Au-Au-200/STAR" / name).read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        return np.array(rows, dtype=float)

    rep = read(f"{OBS}.dat")
    orig = read(f"{OBS}_original_error.dat")
    eta = ETA_EDGES.mean(1)
    mu, var = rep[:, 0], rep[:, 1]            # coluna 'err' guarda a VARIANCIA (bayes_exp.py usa sqrt)
    return eta, mu, np.sqrt(var), np.sqrt(orig[:, 1])


def true_sources(eta, mu, sig):
    fw = np.abs(eta) > 2
    raw = np.stack([
        REL_NORM * mu,                                                     # normalizacao
        S_ASYM * sig * np.sign(eta) * fw,                                  # +frente / -tras
        S_EDGE * sig * np.exp(-((np.abs(eta) - EDGE_CENTER) / EDGE_WIDTH) ** 2) * fw,  # borda
    ], axis=1)                                                             # (n, 3) em unidades de v2
    s = np.linalg.norm(raw, axis=0)
    return raw / s, s                                                      # vhat (unitario), escala rms


def gamma_shape(skew):
    return 4.0 / np.asarray(skew) ** 2


def sample_lambda(rng, s, skew, size):
    k = gamma_shape(skew)
    return s * (rng.gamma(k, 1.0, size=(size, len(k))) - k) / np.sqrt(k)


# --------------------------------------------------------------------------- estimators
def third_cumulant(r):
    rc = r - r.mean(0)
    return np.einsum("ki,kj,kl->ijl", rc, rc, rc) / len(rc)


def symmetric_cp(T, rank, rng, restarts=40, iters=300):
    """Power iteration com reinicios + deflacao, depois refinamento conjunto por minimos quadrados."""
    n = T.shape[0]
    R, comps = T.copy(), []
    for _ in range(rank):
        best = None
        for _ in range(restarts):
            u = rng.normal(size=n); u /= np.linalg.norm(u)
            for _ in range(iters):
                w = np.einsum("ijl,j,l->i", R, u, u)
                if not np.any(w):
                    break
                u = w / np.linalg.norm(w)
            lam = np.einsum("ijl,i,j,l->", R, u, u, u)
            if best is None or abs(lam) > abs(best[0]):
                best = (lam, u)
        comps.append(best)
        R = R - best[0] * np.einsum("i,j,l->ijl", best[1], best[1], best[1])

    p0 = np.concatenate([np.concatenate([[w], u]) for w, u in comps])

    def unpack(p):
        p = p.reshape(rank, n + 1)
        U = p[:, 1:] / np.linalg.norm(p[:, 1:], axis=1, keepdims=True)
        return p[:, 0], U

    def resid(p):
        w, U = unpack(p)
        return (T - np.einsum("a,ai,aj,al->ijl", w, U, U, U)).ravel()

    w, U = unpack(optimize.least_squares(resid, p0).x)
    sgn = np.sign(w)
    return w * sgn, (U.T * sgn)            # pesos > 0, direcoes (n, rank)


def offdiag_scales(r, V):
    """s_a^2 dos elementos fora da diagonal da covariancia das replicas (o estatistico e diagonal)."""
    C = np.cov(r.T)
    iu = np.triu_indices(C.shape[0], 1)
    A = np.stack([np.outer(v, v)[iu] for v in V.T], axis=1)
    s2, _ = optimize.nnls(A, C[iu])
    return np.sqrt(np.maximum(s2, 1e-30))


def estimate_tensor(r, rank, rng):
    w, V = symmetric_cp(third_cumulant(r), rank, rng)
    s = offdiag_scales(r, V)
    return V, s, w / s**3


def estimate_pca(r, rank):
    ev, vec = np.linalg.eigh(np.cov(r.T))  # covariancia completa (analista p=2)
    idx = np.argsort(ev)[::-1][:rank]
    V = vec[:, idx]
    z = (r - r.mean(0)) @ V
    sk = stats.skew(z, axis=0)
    return V * np.sign(sk), np.sqrt(ev[idx]), np.abs(sk)


def match(V_true, V_est):
    """Permutacao que melhor alinha colunas; devolve permutacao e angulos (graus)."""
    from itertools import permutations
    best = None
    for perm in permutations(range(V_true.shape[1])):
        cos = np.abs(np.sum(V_true * V_est[:, list(perm)], axis=0))
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        if best is None or ang.mean() < best[1].mean():
            best = (list(perm), ang)
    return best


# ---------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    eta, mu, sig_tot, sig_orig = load_star()
    V, s = true_sources(eta, mu, sig_tot)
    sys_var = (V**2 * s**2).sum(1)
    share = sys_var / sig_tot**2
    assert share.max() < 0.9, f"fontes excedem 90% da variancia informada: {share.max():.2f}"
    sig_stat = np.sqrt(sig_tot**2 - sys_var)

    np.savetxt(DATA / "star_v2_eta.csv", np.column_stack([ETA_EDGES, eta, mu, sig_tot, sig_orig, sig_stat]),
               delimiter=",", header="eta_low,eta_high,eta,v2,sigma_total,sigma_original,sigma_stat_toy",
               comments="", fmt="%.6g")
    np.savez(DATA / "true_sources.npz", eta=eta, v_hat=V, s=s, skew=SKEW, k=gamma_shape(SKEW),
             sigma_total=sig_tot, sigma_stat=sig_stat, names=NAMES)

    # ---- Etapa 1: replicas (variacoes sistematicas + flutuacao estatistica)
    lam = sample_lambda(rng, s, SKEW, args.replicas)
    reps = lam @ V.T + rng.normal(0, sig_stat, size=(args.replicas, len(eta)))
    np.savez(DATA / "replicas.npz", replicas=reps, lam=lam, seed=args.seed)

    # ---- Etapa 2: Tensor PCA (p=3) vs PCA (p=2)
    Vt, st, gt = estimate_tensor(reps, 3, rng)
    perm, ang_t = match(V, Vt)
    Vt, st, gt = Vt[:, perm], st[perm], gt[perm]
    Vt = Vt * np.sign(np.sum(Vt * V, axis=0))          # mesma orientacao das verdadeiras (so para comparar)
    Vp, sp, gp = estimate_pca(reps, 3)
    perm_p, ang_p = match(V, Vp)

    # recuperacao vs numero de replicas
    Ns = [100, 300, 1000, 3000, 10000]
    curve = {"N": Ns, "tensor_deg": [], "pca_deg": []}
    for N in Ns:
        at, ap_ = [], []
        for rep in range(5):
            rr = np.random.default_rng(10_000 * N + rep)
            r = sample_lambda(rr, s, SKEW, N) @ V.T + rr.normal(0, sig_stat, size=(N, len(eta)))
            at.append(match(V, estimate_tensor(r, 3, rr)[0])[1].mean())
            ap_.append(match(V, estimate_pca(r, 3)[0])[1].mean())
        curve["tensor_deg"].append([float(np.mean(at)), float(np.std(at))])
        curve["pca_deg"].append([float(np.mean(ap_)), float(np.std(ap_))])

    # entrada da cadeia: fontes INFERIDAS; estatistico = total - sistematico inferido
    sys_inf = (Vt**2 * st**2).sum(1)
    sig_stat_inf = np.sqrt(np.maximum(sig_tot**2 - sys_inf, (0.05 * sig_tot) ** 2))
    np.savez(DATA / "inferred_sources.npz", eta=eta, v_hat=Vt, s=st, skew=gt, k=gamma_shape(gt),
             sigma_total=sig_tot, sigma_stat=sig_stat_inf, observable=OBS, n_replicas=args.replicas)
    report = dict(
        observable=OBS, n_bins=len(eta), n_replicas=args.replicas, seed=args.seed,
        variance_share_systematic=dict(min=float(share.min()), max=float(share.max())),
        sources=[dict(name=NAMES[a], s_true=float(s[a]), s_tensor=float(st[a]), skew_true=float(SKEW[a]),
                      skew_tensor=float(gt[a]), k_tensor=float(gamma_shape(gt[a])), angle_tensor_deg=float(ang_t[a]),
                      angle_pca_deg=float(ang_p[a])) for a in range(3)],
        recovery_vs_N=curve,
    )
    (DATA / "inferred_sources.json").write_text(json.dumps(report, indent=2))

    # ---- figuras
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), sharey=False)
    for a in range(3):
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
    ax.set_title("STAR v2(eta): identificacao das fontes"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "recovery_vs_N.png", dpi=160)

    print(json.dumps({k: report[k] for k in ("variance_share_systematic", "sources")}, indent=1))
    print(json.dumps(curve))


if __name__ == "__main__":
    main()
