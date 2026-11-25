"""R1: parte impar em eta do sistematico do PHOBOS dN/deta (Au+Au 200), via simetria eta <-> -eta.

A[c,j] = (y(eta_j) - y(-eta_j)) / (y(eta_j) + y(-eta_j)),  c = 11 classes, j = 27 pares (|eta| = 0.1 ... 5.3)
Em Au+Au a fisica e simetrica => A e sistematico (parte impar) + ruido.
- Amplitude por regiao (Octagon |eta|<3.2, Rings |eta|>3.2) vs incerteza citada
- Decomposicao SVD da matriz A (11 x 27): modo comum entre classes? (2o momento: correlacao entre classes e entre bins)
- Correlacao entre bins vizinhos em eta (suavidade => sistematico, nao ruido)
- 3o momento: assimetria das amplitudes do modo dominante e dos residuos (indicativo)
"""
import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "HIC_experimental_data", "Au-Au-200", "PHOBOS")
files = sorted(glob.glob(os.path.join(D, "dNdeta_eta_cen_*_PHOB.dat")))
eta_c = np.round(np.arange(-5.3, 5.31, 0.2), 2)          # 54 centros
classes, Y, S = [], [], []
for f in files:
    m = re.search(r"cen_(\d+)_(\d+)", f); classes.append(f"{int(m.group(1))}-{int(m.group(2))}%")
    L = np.array([list(map(float, l.split())) for l in open(f) if not l.startswith("#")])
    Y.append(L[:, 0]); S.append(np.sqrt(L[:, 1]))          # coluna err = variancia
Y, S = np.array(Y), np.array(S)
n = Y.shape[1]; npair = n // 2
pos = Y[:, npair:]; neg = Y[:, :npair][:, ::-1]           # pares |eta| crescente
absEta = eta_c[npair:]
A = (pos - neg) / (pos + neg)                            # 11 x 27
rel = 0.5 * (S[:, npair:] / pos + S[:, :npair][:, ::-1] / neg)   # incerteza citada relativa
octa = absEta < 3.2; rings = ~octa

def rms(x): return float(np.sqrt(np.mean(np.square(x))))
res = dict(classes=classes, abs_eta=absEta.tolist())
res["amplitude"] = {reg: dict(rms_A_pct=100 * rms(A[:, msk]), quoted_rel_pct=100 * float(rel[:, msk].mean()),
                              ratio=rms(A[:, msk]) / float(rel[:, msk].mean()))
                    for reg, msk in (("Octagon", octa), ("Rings", rings), ("total", np.ones_like(octa)))}

# SVD: A = sum_k s_k u_k v_k^T  (u: classes, v: eta)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
frac = s ** 2 / np.sum(s ** 2)
res["svd_variance_fraction"] = frac[:5].tolist()
v1 = Vt[0] * np.sign(Vt[0][rings].sum() or 1); u1 = U[:, 0] * s[0] * np.sign(Vt[0][rings].sum() or 1)
res["mode1_eta_profile"] = v1.tolist(); res["mode1_class_amplitude"] = u1.tolist()

# perfil medio e correlacao entre classes
Abar = A.mean(0)
C = np.corrcoef(A)                                        # 11x11 correlacao entre classes (ao longo de eta)
iu = np.triu_indices(len(classes), 1)
res["mean_corr_between_classes"] = float(C[iu].mean())
res["mean_corr_between_classes_octagon"] = float(np.corrcoef(A[:, octa])[iu].mean())
res["mean_corr_between_classes_rings"] = float(np.corrcoef(A[:, rings])[iu].mean())
# correlacao entre bins vizinhos em eta (lag 1), media sobre classes
lag1 = [np.corrcoef(a[:-1], a[1:])[0, 1] for a in A]
res["lag1_corr_eta_mean"] = float(np.mean(lag1))

# 3o momento (indicativo)
def skew(x): x = np.asarray(x) - np.mean(x); return float(np.mean(x ** 3) / np.mean(x ** 2) ** 1.5)
resid = A - np.outer(U[:, 0] * s[0], Vt[0])
rng = np.random.default_rng(2)
def bs_sd(x, B=2000): x = np.ravel(x); return float(np.std([skew(rng.choice(x, len(x))) for _ in range(B)]))
res["third_moment"] = dict(skew_all_A=skew(A.ravel()), skew_all_A_sd=bs_sd(A),
                           skew_A_rings=skew(A[:, rings].ravel()), skew_A_rings_sd=bs_sd(A[:, rings]),
                           skew_A_octagon=skew(A[:, octa].ravel()), skew_A_octagon_sd=bs_sd(A[:, octa]),
                           skew_resid=skew(resid.ravel()), skew_resid_sd=bs_sd(resid),
                           skew_mode1_amplitudes=skew(u1), n_mode1=len(u1))
json.dump(res, open(os.path.join(HERE, "R1_results.json"), "w"), indent=1)

# figuras
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, c in zip(A, classes): ax[0].plot(absEta, 100 * a, "-", lw=0.8, alpha=0.7, label=c)
ax[0].plot(absEta, 100 * Abar, "k-", lw=2.2, label="média")
ax[0].fill_between(absEta, -100 * rel.mean(0), 100 * rel.mean(0), color="0.88", zorder=0, label="± incerteza citada")
ax[0].axvline(3.2, color="C3", ls="--", lw=1); ax[0].text(3.25, ax[0].get_ylim()[1] * 0.85 if ax[0].get_ylim()[1] else 5, "Rings", color="C3", fontsize=8)
ax[0].set_ylim(-12, 12); ax[0].set_xlabel("|η|"); ax[0].set_ylabel("A(η) = [y(η)−y(−η)]/[y(η)+y(−η)]  (%)")
ax[0].set_title("Assimetria η↔−η por classe"); ax[0].legend(fontsize=6, ncol=2)
ax[1].plot(absEta, 100 * A.mean(0), "k-", lw=2, label="média")
ax[1].fill_between(absEta, 100 * (A.mean(0) - A.std(0)), 100 * (A.mean(0) + A.std(0)), color="C0", alpha=0.3, label="±1 desvio entre classes")
ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(3.2, color="C3", ls="--", lw=1)
ax[1].set_xlabel("|η|"); ax[1].set_title("Zoom: média e dispersão entre classes"); ax[1].legend(fontsize=7)
im = ax[2].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r"); ax[2].set_xticks(range(len(classes))); ax[2].set_xticklabels(classes, rotation=90, fontsize=6)
ax[2].set_yticks(range(len(classes))); ax[2].set_yticklabels(classes, fontsize=6); ax[2].set_title("Correlação de A(η) entre classes"); fig.colorbar(im, ax=ax[2], fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "R1_simetria.png"), dpi=130); fig.savefig(os.path.join(HERE, "R1_simetria.pdf"))
print(json.dumps({k: v for k, v in res.items() if k not in ("mode1_eta_profile", "abs_eta")}, indent=1))
