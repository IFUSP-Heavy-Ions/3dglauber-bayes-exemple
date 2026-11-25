# Quantum Tensor PCA on a simulator (Qiskit)

This is an isolated study. It reads the detector-split replicas in `../data/detector_split/` and writes only to `output/`.

## What runs on the quantum circuit

The tensor power method finds each systematic source as a fixed point of

    u  <-  top eigenvector of  M(u) = kappa3(., ., u)

The **spectral step** (the top eigenvector) is done in three ways. Everything else is identical: the same third cumulant, deflation, random restarts, least-squares refinement and estimation of `s_a` and `skew_a`.

| solver | what it is |
|---|---|
| `classical` | `numpy.linalg.eigh` |
| `qpe_sv` | quantum phase estimation in Qiskit Aer, exact statevector readout (ideal amplitude access) |
| `qpe_shots` | the same circuit with finite shots. `|u_i|^2` comes from the counts after post-selecting the top-eigenvalue clock outcome. The signs come from interference circuits (a Hadamard on the pair `{i, j0}`). |

**Circuit:**

- `M` is padded to 32×32, which needs 5 system qubits.
- It is rescaled so that its eigenvalues lie in [−1, 1] and mapped to phases in [0, 0.9]. Then `U = exp(2πi H)`.
- The current `u` is prepared on the system register (`StatePreparation`).
- `m` clock qubits apply controlled-`U^(2^k)` followed by the inverse QFT.
- Post-selecting the highest-phase clock peak projects `|u>` onto the top eigenvector.

With `m = 9` the circuit has 14 qubits.

This is the spectral primitive of Hastings' quantum Tensor PCA, Quantum 4, 237 (2020): phase estimation on a Hamiltonian built from the tensor, here at the lowest level. At n = 21 it is a **correctness demonstration**, not a speedup.

## Does it have to give the same result?

Not bit for bit. The quantum result must agree with the classical one **within the statistical uncertainty of the estimate itself**, which is measured by bootstrapping the 5000 replicas. The differences come from three places:

- **QPE resolution:** a finite number of clock qubits leaks some weight from nearby eigenvalues. The error falls as the number of clock qubits grows (self-test: 15° with 7 qubits, 2° with 9).
- **Shot noise:** the amplitudes are estimated from counts, with an error of about `1/sqrt(shots after post-selection)`.
- **Post-selection:** the cost is `1/p_post`. When the iteration starts from a random `u` the overlap with the top eigenvector is small. Hastings' algorithm uses amplitude amplification here.

## Run (server)

```bash
python3.12 -m venv .venv && .venv/bin/pip install "qiskit>=1.2" qiskit-aer numpy scipy matplotlib pylatexenc
.venv/bin/python quantum_tensor_pca.py --selftest --clock 9
.venv/bin/python quantum_tensor_pca.py --clock 9 --shots 8192 --restarts 8 --bootstrap 20
```

Tested with Qiskit 2.6.2, Aer 0.18.2 and Python 3.12.

## Output (`output/`)

- `results.json`: angles between quantum and classical, angles to the truth, `s` and skewness, bootstrap tolerance, timing, number of circuits.
- `quantum_vs_classical_sources.png/pdf`: true sources compared with the classical, statevector and shots results.
- `qpe_circuit.png/pdf`: one spectral step, for the poster.
- `inferred_sources_{classical,qpe_sv,qpe_shots}.npz`: same format as `../data/detector_split/inferred_sources.npz`. They can be passed to `bayes_mcmc_nuisance.py --sources ...`.
