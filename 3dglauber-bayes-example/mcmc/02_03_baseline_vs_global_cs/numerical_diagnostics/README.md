# Numerical baseline versus global CS diagnostics

These results adapt the numerical parts of the hic-bayes AllDetectorsAuAu200
`diagnose_covariance_models.py` and `diagnose_uncertainty_correlation.py` analyses
to the joint Au–Au-200 + d–Au-200 JETSCAPE studies. No MCMC, training or plots
are generated. The comparison notebook embeds the main numerical tables and
interpretations; `report.md` contains the same numerical section.

From the project directory, run:

```sh
python mcmc/scripts/diagnose_phase2_numerical.py
```

To rebuild just the notebook section from these saved results:

```sh
python mcmc/scripts/diagnose_phase2_numerical.py --refresh-notebook
```

Dependencies: numpy, scipy, h5py, pandas, dill, scikit-learn, joblib and
threadpoolctl. Python 3.14.3 / scikit-learn 1.8.0 was used for this snapshot;
the emulator caches were saved with scikit-learn 1.6.1. All 80 saved observable
quantile comparisons matched (rtol 1e-6, atol 1e-10), which validates agreement
of predicted means with the plot results, not historical predictive covariance.
Exact package versions and source/input hashes are in `diagnostics.json`.

## Files

- `run_inventory.csv`: chain layout, stride and retained sample counts.
- `posterior_parameter_summary.csv`: means, standard deviations and quantiles.
- `posterior_sensitivity.csv`, `parameter_sensitivity_ranking.csv`: JS divergence
  in bits, Wasserstein distance, normalized median shifts and 68%/90% width ratios.
- `parameter_pair_correlations.csv`: baseline and CS Pearson parameter correlations.
- `covariance_metrics.csv`: ranks, eigenvalues, trace, norms, conditioning and
  determinant for experimental, mean emulator and total covariance matrices.
- `covariance_vs_baseline.csv`: experimental covariance differences.
- `uncertainty_summary.csv`, `uncertainty_by_observable.csv`: posterior-mean
  variation relative to experimental standard deviations, and CV comparisons.
- `posterior_predictive_per_bin.csv`: per-bin physical-space predictions and
  uncertainty ratios, plus likelihood-space residual diagnostics.
- `posterior_predictive_summary.csv`: Gaussian terms at the coordinatewise
  median prediction with posterior-averaged emulator covariance.
- `cross_covariance_likelihoods.csv`: Gaussian terms for both covariance models
  at the same saved posterior draws, including sample-dependent emulator covariance.
- `time_window_intervals.csv`: central 60%/90% sample intervals in ten time windows.
- `saved_quantile_validation.csv`: comparisons against the saved observable CSVs.
- `*_diagnostic_arrays.npz`: predictions and covariance arrays used in the analysis.

The experimental CS matrix at rho=1 is rank one. No jitter, pseudoinverse or
pseudo-determinant is used to turn it into a standalone Gaussian density.
Gaussian calculations use experimental plus emulator covariance and require
Cholesky factorization to succeed. Log likelihoods here are not evidences,
Bayes factors or convergence tests. Normalized median shifts use the square
root of the sum of parameter variances, matching the source uncertainty script;
they differ from the earlier notebook's baseline-90%-width normalization.
