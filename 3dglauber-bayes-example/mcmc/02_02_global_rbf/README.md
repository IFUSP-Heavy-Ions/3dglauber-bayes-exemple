# 02_02 · RBF global (common physical lengths)

Experimental covariance: every bin of a system is a point (pT, eta, centrality);
`R_ij = prod_axis exp(-((x_i-x_j)/L_axis)^2)`, L_pt = 0.5 GeV/c, L_eta = 1, L_cent = 10 (same as RBF sector).
Neutral cross-observable factor (1). Au-Au and d-Au uncorrelated. Code: `experimental_global_covariance.py`,
`bayes_mcmc_rbf_global.py`, `bayes_mcmc_rbf_global_chain.py`; acceptance table `global_rbf_acceptance.json`
(integrated axis = midpoint of the acceptance window; "all pT" = [0, 1.5] GeV/c; windows from the original papers).
Method notes: journal `2026-11-23-JETSCAPE/img/RBF_GLOBAL_METHOD.md`.

## Run (24/11/2026)
- Check: `python bayes_mcmc_rbf_global.py --rbf-length-pt 0.5 --rbf-length-eta 1 --rbf-length-centrality 10 --check-covariance`
  → PSD in both systems; Au-Au 791 bins (rank 542, mean off-diagonal r = 0.047), d-Au 126 bins (rank 79, mean r = 0.19).
- Submitted: `condor_submit condor_rbf_global.sub` (200 walkers, burn-in 20 000, production 20 000, 20 CPUs).
  - cluster 5077: failed at the first parallel step (Chain class not picklable) → fixed by moving it to
    `bayes_mcmc_rbf_global_chain.py`; its empty HDF was removed, logs in `logs/failed_5077/`.
  - **cluster 5078: running on qgp1** (started 24/11/2026 ~22:15).
- Output: `chain-idf-0-rbf-global-<signature12>.hdf` in this folder. The signature printed by the local
  `--check-covariance` (00950d99d3b5) differs from the chain's (c33aba121315), as in the sector studies
  (`covariance_signature_matches = False`): the check rebuilds sigmas from the data files, the chain uses
  `bayes_exp.Y_exp_data`, and numpy versions differ. The chain's own attributes are authoritative.
- When finished, set `common.MODELS["rbf_global"]["path"]` in `mcmc/scripts/paper_figures/common.py` and rerun the
  `global` figure commands (README there).
