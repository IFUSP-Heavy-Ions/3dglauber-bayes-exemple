#!/usr/bin/env python3
"""Tensor PCA of the STAR v2(eta) systematic replicas with a quantum spectral step (Qiskit).

Isolated study: reads ../data/detector_split/{replicas,true_sources,inferred_sources}.npz
and writes only to ./output/. Nothing else in the repository is used or changed except
the helper functions imported from ../toy_model/sources.py.

Algorithm (same for every solver; only the spectral step changes)
    kappa3 = third cumulant of the replicas (n = 21 bins)
    for each source (deflation), several random restarts:
        u <- top eigenvector of M(u) = kappa3(., ., u)       <- SPECTRAL STEP
        until u stops changing;  keep the restart with largest |kappa3(u,u,u)|
        deflate kappa3 <- kappa3 - w u x u x u
    (optional) joint least-squares refinement, as in sources.py
A fixed point satisfies kappa3(., u, u) = w u: a tensor eigenvector, the same object
found by the tensor power iteration of sources.py.

Spectral step, three implementations
    classical : numpy.linalg.eigh
    qpe_sv    : quantum phase estimation in Qiskit Aer, exact statevector readout
                (ideal amplitude access)
    qpe_shots : the same circuit with finite shots; |u_i|^2 from counts after
                post-selecting the top-eigenvalue clock outcome, signs from
                interference circuits that apply a Hadamard on the pair {i, j0}
QPE: M is padded to 32 x 32 (5 system qubits), rescaled by its Frobenius norm so
that all eigenvalues lie in [-1, 1], mapped to phases phi = 0.45 (lambda + 1) in
[0, 0.9], U = exp(2 pi i H). m clock qubits; controlled-U^(2^k) are 6-qubit unitaries.
The state |u> is prepared on the system register; post-selecting the highest-phase
clock outcome projects it onto the top eigenvector (up to QPE leakage).

This is the spectral primitive of Hastings' quantum Tensor PCA (phase estimation on a
Hamiltonian built from the tensor) at the lowest level; it is a correctness
demonstration on a simulator, not a speedup at n = 21.

Usage (from study_nongaussian_systematics/quantum, server venv):
    .venv/bin/python quantum_tensor_pca.py --selftest
    .venv/bin/python quantum_tensor_pca.py [--clock 7] [--shots 8192] [--restarts 8] [--bootstrap 20]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg, optimize

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "toy_model"))
from sources import gamma_shape, match, offdiag_scales, third_cumulant  # noqa: E402

DATA = STUDY / "data" / "detector_split"
OUT = HERE / "output"
NAMES = ["norm. TPC", "norm. FTPC", "asymmetry", "edge"]
NSYS = 5                      # 2^5 = 32 >= 21 bins


# ------------------------------------------------------------------ quantum spectral step
class QPEEigensolver:
    def __init__(self, clock=7, shots=None, seed=1):
        from qiskit_aer import AerSimulator
        self.m = clock
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self.sim = AerSimulator(method="statevector", seed_simulator=seed,
                                max_parallel_experiments=0, max_parallel_threads=0)
        self.calls = 0
        self.circuits = 0
        self.last = {}

    def _unitaries(self, M):
        n = M.shape[0]
        Mp = np.zeros((2 ** NSYS, 2 ** NSYS))
        Mp[:n, :n] = M / np.linalg.norm(M)                      # eigenvalues in [-1, 1]
        H = 0.45 * (Mp + np.eye(2 ** NSYS))                     # phases in [0, 0.9]
        U = linalg.expm(2j * np.pi * H)
        P0, P1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
        I = np.eye(2 ** NSYS)
        cus, Uk = [], U
        for _ in range(self.m):
            cus.append(np.kron(I, P0) + np.kron(Uk, P1))         # qargs [clock_k, sys...]
            Uk = Uk @ Uk
        return cus

    @staticmethod
    def _top_bin(pk, floor):
        """Highest-phase clock outcome that is a local maximum above `floor`.

        QPE leakage tails decay monotonically away from each eigenphase, so they do not
        create local maxima; the top eigenvalue shows up as the highest-phase peak even
        when its overlap with |u> is small."""
        ext = np.concatenate([[0.0], pk, [0.0]])
        peak = (pk >= ext[:-2]) & (pk >= ext[2:]) & (pk >= floor)
        return int(np.flatnonzero(peak).max())

    def _circuit(self, cus, u, readout=None, measure=True):
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
        from qiskit.circuit.library import QFTGate, StatePreparation, UnitaryGate
        clk, sysq = QuantumRegister(self.m, "clock"), QuantumRegister(NSYS, "sys")
        qc = QuantumCircuit(clk, sysq)
        amp = np.zeros(2 ** NSYS); amp[:len(u)] = u
        qc.append(StatePreparation(amp / np.linalg.norm(amp)), sysq)
        qc.h(clk)
        for k, cu in enumerate(cus):
            qc.append(UnitaryGate(cu, label=f"cU^{2**k}"), [clk[k], *sysq])
        qc.append(QFTGate(self.m).inverse(), clk)
        if readout is not None:
            qc.append(UnitaryGate(readout, label="R"), sysq)
        if measure:
            cc, cs = ClassicalRegister(self.m, "c"), ClassicalRegister(NSYS, "s")
            qc.add_register(cc); qc.add_register(cs)
            qc.measure(clk, cc); qc.measure(sysq, cs)
        else:
            qc.save_statevector()
        return qc

    def _run(self, circs, shots=None):
        from qiskit import transpile
        tc = transpile(circs, self.sim, optimization_level=0)
        self.circuits += len(circs)
        return self.sim.run(tc, shots=shots).result()

    @staticmethod
    def _counts(result, i):
        out = {}
        for key, c in result.get_counts(i).items():
            s, k = key.split()                                   # "sysbits clockbits"
            out[(int(k, 2), int(s, 2))] = out.get((int(k, 2), int(s, 2)), 0) + c
        return out

    def top_eigenvector(self, M, u):
        self.calls += 1
        n = M.shape[0]
        cus = self._unitaries(M)
        if self.shots is None:                                   # ideal readout
            res = self._run([self._circuit(cus, u, measure=False)])
            psi = np.asarray(res.get_statevector(0)).reshape(2 ** NSYS, 2 ** self.m)  # [sys, clock]
            pk = (np.abs(psi) ** 2).sum(0)
            k = self._top_bin(pk, 1e-4)
            v = psi[:, k]
            v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
            v = v.real[:n]
            self.last = dict(bin=int(k), p_post=float(pk[k]), leak=float(np.linalg.norm(psi[n:, k]) ** 2 / pk[k]))
            return v / np.linalg.norm(v)
        # finite shots
        res = self._run([self._circuit(cus, u)], shots=self.shots)
        cnt = self._counts(res, 0)
        pk = np.zeros(2 ** self.m)
        for (k, s), c in cnt.items():
            pk[k] += c
        k = self._top_bin(pk, 10)
        mag2 = np.zeros(2 ** NSYS)
        for (kk, s), c in cnt.items():
            if kk == k:
                mag2[s] += c
        mag2 = mag2[:n] / max(mag2.sum(), 1)
        j0 = int(np.argmax(mag2))
        # signs: Hadamard on the pair {i, j0}; amplitudes (v_i +- v_j0)/sqrt2
        circs, idx = [], [i for i in range(n) if i != j0 and mag2[i] > 0]
        for i in idx:
            R = np.eye(2 ** NSYS)
            R[np.ix_([i, j0], [i, j0])] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            circs.append(self._circuit(cus, u, readout=R))
        sign = np.ones(n)
        if circs:
            res = self._run(circs, shots=self.shots)
            for c_i, i in enumerate(idx):
                cnt = self._counts(res, c_i)
                pi, pj = cnt.get((k, i), 0), cnt.get((k, j0), 0)
                sign[i] = 1.0 if pi >= pj else -1.0
        v = sign * np.sqrt(mag2)
        self.last = dict(bin=int(k), p_post=float(pk[k] / pk.sum()), shots_post=int(pk[k]))
        return v / np.linalg.norm(v)


def classical_top_eigenvector(M, u):
    w, V = np.linalg.eigh(M)
    return V[:, -1]


# ------------------------------------------------------------------------ Tensor PCA
def tensor_pca(T, rank, solver, rng, restarts=8, iters=25, tol=1e-6, log=None):
    n = T.shape[0]
    R, comps, history = T.copy(), [], []
    starts = [[rng.normal(size=n) for _ in range(restarts)] for _ in range(rank)]
    for a in range(rank):
        best = None
        for r0 in starts[a]:
            u = r0 / np.linalg.norm(r0)
            for it in range(iters):
                M = np.einsum("ijl,l->ij", R, u)
                v = solver(M, u)
                v = v * np.sign(v @ u if v @ u != 0 else 1.0)
                delta = 1 - abs(v @ u)
                u = v
                if delta < tol:
                    break
            lam = np.einsum("ijl,i,j,l->", R, u, u, u)
            history.append(dict(source=a, iterations=it + 1, w=float(lam), last_delta=float(delta)))
            if best is None or abs(lam) > abs(best[0]):
                best = (lam, u)
        comps.append(best)
        R = R - best[0] * np.einsum("i,j,l->ijl", best[1], best[1], best[1])
        if log:
            log(f"  source {a}: w = {best[0]:.3e}")
    w = np.array([c[0] for c in comps]); U = np.stack([c[1] for c in comps], 1)
    s = np.sign(w)
    return w * s, U * s, history


def refine(T, w, U):
    """Joint least-squares CP fit. T is rescaled to unit norm first: with entries ~1e-7
    the cost is ~1e-14 and scipy's gtol is met at the starting point. Weights restart
    from T(u,u,u) of the full tensor (the deflation weights refer to residual tensors)."""
    scale = np.linalg.norm(T)
    T = T / scale
    w = np.einsum("ijl,ia,ja,la->a", T, U, U, U)
    rank, n = len(w), U.shape[0]
    p0 = np.concatenate([np.concatenate([[w[a]], U[:, a]]) for a in range(rank)])

    def unpack(p):
        p = p.reshape(rank, n + 1)
        V = p[:, 1:] / np.linalg.norm(p[:, 1:], axis=1, keepdims=True)
        return p[:, 0], V

    def resid(p):
        ww, V = unpack(p)
        return (T - np.einsum("a,ai,aj,al->ijl", ww, V, V, V)).ravel()

    ww, V = unpack(optimize.least_squares(resid, p0).x)
    s = np.sign(ww)
    return ww * s * scale, (V.T * s)


def summarize(r, w, U, V_true):
    perm, ang = match(V_true, U)
    U, w = U[:, perm], w[perm]
    U = U * np.sign(np.sum(U * V_true, axis=0))
    s = offdiag_scales(r, U)
    return dict(V=U, w=w, s=s, skew=w / s ** 3, angle_to_truth=ang)


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(a * b, axis=0)), -1, 1)))


# ------------------------------------------------------------------------------- main
def selftest(clock):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(21, 21)); M = A + A.T
    u = rng.normal(size=21); u /= np.linalg.norm(u)
    ref = classical_top_eigenvector(M, u)
    for shots in (None, 8192):
        q = QPEEigensolver(clock=clock, shots=shots)
        v = q.top_eigenvector(M, u)
        print(f"selftest shots={shots}: angle to eigh = {angle(v, ref):.3f} deg, info={q.last}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clock", type=int, default=7, help="QPE clock qubits")
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--bootstrap", type=int, default=20, help="classical bootstrap resamples")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.clock); return
    OUT.mkdir(exist_ok=True)

    reps = np.load(DATA / "replicas.npz")["replicas"]
    truth = np.load(DATA / "true_sources.npz")
    orig = np.load(DATA / "inferred_sources.npz")
    V_true, rank = truth["v_hat"], truth["v_hat"].shape[1]
    r = reps - reps.mean(0)
    T = third_cumulant(reps)

    solvers = {"classical": classical_top_eigenvector,
               "qpe_sv": QPEEigensolver(clock=args.clock, shots=None, seed=args.seed),
               "qpe_shots": QPEEigensolver(clock=args.clock, shots=args.shots, seed=args.seed)}
    results, timing = {}, {}
    for name, solver in solvers.items():
        t0 = time.time()
        fn = solver if callable(solver) and not isinstance(solver, QPEEigensolver) else solver.top_eigenvector
        print(f"[{name}]", flush=True)
        w, U, hist = tensor_pca(T, rank, fn, np.random.default_rng(args.seed), args.restarts, args.iters,
                                log=lambda s: print(s, flush=True))
        raw = summarize(r, w, U, V_true)
        wr, Ur = refine(T, w, U)
        ref = summarize(r, wr, Ur, V_true)
        results[name] = dict(raw=raw, refined=ref, history=hist)
        timing[name] = dict(seconds=time.time() - t0,
                            spectral_calls=getattr(solver, "calls", None), circuits=getattr(solver, "circuits", None))
        print(f"  {timing[name]}", flush=True)

    # statistical tolerance: bootstrap of the classical estimate
    rng = np.random.default_rng(args.seed + 1)
    boot = []
    for b in range(args.bootstrap):
        rb = reps[rng.integers(0, len(reps), len(reps))]
        wb, Ub, _ = tensor_pca(third_cumulant(rb), rank, classical_top_eigenvector, np.random.default_rng(args.seed), args.restarts, args.iters)
        sb = summarize(rb - rb.mean(0), wb, Ub, V_true)
        boot.append(angle(sb["V"], results["classical"]["raw"]["V"]))
    boot = np.array(boot)

    # comparison
    table = {}
    for stage in ("raw", "refined"):
        c = results["classical"][stage]
        for q in ("qpe_sv", "qpe_shots"):
            x = results[q][stage]
            table[f"{q}_{stage}"] = dict(
                angle_to_classical_deg=angle(x["V"], c["V"]).tolist(),
                angle_to_truth_deg=x["angle_to_truth"].tolist(),
                s_rel_diff=(x["s"] / c["s"] - 1).tolist(),
                skew=x["skew"].tolist())
        table[f"classical_{stage}"] = dict(angle_to_truth_deg=c["angle_to_truth"].tolist(),
                                           s=c["s"].tolist(), skew=c["skew"].tolist())
    table["original_power_iteration"] = dict(angle_to_truth_deg=angle(orig["v_hat"], V_true).tolist(),
                                             angle_to_classical_raw_deg=angle(orig["v_hat"], results["classical"]["raw"]["V"]).tolist())
    table["bootstrap_classical_angle_deg"] = dict(median=np.median(boot, 0).tolist(), q84=np.quantile(boot, .84, 0).tolist())
    report = dict(config=vars(args), names=NAMES, n_bins=int(T.shape[0]), n_replicas=int(len(reps)),
                  qubits=dict(system=NSYS, clock=args.clock, total=NSYS + args.clock),
                  timing=timing, comparison=table)
    (OUT / "results.json").write_text(json.dumps(report, indent=1))
    for name in results:
        x = results[name]["refined"]
        np.savez(OUT / f"inferred_sources_{name}.npz", v_hat=x["V"], s=x["s"], skew=x["skew"], k=gamma_shape(x["skew"]),
                 sigma_total=orig["sigma_total"], eta=orig["eta"], observable=str(orig["observable"]),
                 labels=np.array(["lambda_norm_TPC", "lambda_norm_FTPC", "lambda_asym", "lambda_edge"]),
                 sigma_stat=np.sqrt(np.maximum(orig["sigma_total"] ** 2 - (x["V"] ** 2 * x["s"] ** 2).sum(1),
                                               (0.05 * orig["sigma_total"]) ** 2)))
    np.savez(OUT / "vectors.npz", eta=orig["eta"], truth=V_true * truth["s"],
             **{f"{n}_{st}": results[n][st]["V"] * results[n][st]["s"] for n in results for st in ("raw", "refined")})

    print("\nangle to classical (raw):")
    for q in ("qpe_sv", "qpe_shots"):
        print(f"  {q:10s}", np.round(table[f'{q}_raw']['angle_to_classical_deg'], 2))
    print("bootstrap spread of classical (median):", np.round(table["bootstrap_classical_angle_deg"]["median"], 2))
    print("angle to truth (raw): classical", np.round(results["classical"]["raw"]["angle_to_truth"], 2),
          "| sv", np.round(results["qpe_sv"]["raw"]["angle_to_truth"], 2),
          "| shots", np.round(results["qpe_shots"]["raw"]["angle_to_truth"], 2))
    plot(results, V_true * truth["s"], orig["eta"], table, args)
    draw_circuit(args.clock)


def plot(results, truth, eta, table, args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    sty = {"classical": ("o", "#2e86ab", "classical eigh"), "qpe_sv": ("s", "#2a9d5c", "QPE statevector"),
           "qpe_shots": ("x", "#d1495b", f"QPE {args.shots} shots")}
    for a in range(4):
        ax[a].bar(eta, truth[:, a] * 1e3, width=0.17, color="#ccc", label="true source")
        for name, (mk, c, lab) in sty.items():
            x = results[name]["raw"]
            ax[a].plot(eta, x["V"][:, a] * x["s"][a] * 1e3, mk, color=c, ms=6 if mk != "x" else 8, mfc="none" if mk == "s" else c, label=lab)
        d = table["qpe_shots_raw"]["angle_to_classical_deg"][a]; dsv = table["qpe_sv_raw"]["angle_to_classical_deg"][a]
        bs = table["bootstrap_classical_angle_deg"]["median"][a]
        ax[a].set_title(f"{NAMES[a]}\nquantum vs classical: sv {dsv:.2f}°, shots {d:.2f}°\nstatistical (bootstrap) {bs:.2f}°", fontsize=9)
        ax[a].set_xlabel(r"$\eta$")
    ax[0].set_ylabel(r"$s_a\hat v_a\ (\times10^{-3})$"); ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "quantum_vs_classical_sources.png", dpi=160); fig.savefig(OUT / "quantum_vs_classical_sources.pdf")


def draw_circuit(clock):
    """Circuit diagram of one spectral step (for the poster)."""
    import matplotlib; matplotlib.use("Agg")
    q = QPEEigensolver(clock=clock, shots=1)
    M = np.eye(21); u = np.ones(21) / np.sqrt(21)
    qc = q._circuit(q._unitaries(M), u)
    try:
        fig = qc.draw("mpl", fold=-1, style="clifford")
        fig.savefig(OUT / "qpe_circuit.png", dpi=160, bbox_inches="tight"); fig.savefig(OUT / "qpe_circuit.pdf", bbox_inches="tight")
    except Exception as e:                                      # pylatexenc missing etc.
        (OUT / "qpe_circuit.txt").write_text(str(qc.draw("text", fold=-1)))
        print("circuit drawn as text:", e)


if __name__ == "__main__":
    main()
