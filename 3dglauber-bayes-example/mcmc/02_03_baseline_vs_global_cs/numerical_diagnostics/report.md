## Numerical covariance and uncertainty diagnostics

This section adapts the numerical analyses in `diagnose_covariance_models.py` and `diagnose_uncertainty_correlation.py` from the hic-bayes AllDetectorsAuAu200 study to the saved **joint Au–Au + d–Au** baseline and CS runs. The parameters are shared by the joint fit; parameter sensitivity is therefore reported once, not separately for each system. All new outputs are numerical tables, with full CSVs in `numerical_diagnostics/`.

### Definitions and reproducibility

Posterior comparisons use up to 200,000 samples per chain (saved-step stride 50, no extra burn-in). Jensen–Shannon divergence uses 80 shared bins over the prior range and a 10⁻¹² count pseudocount, as in the original script. It is reported in bits (0 means identical binned marginals; 1 is maximal separation). Wasserstein distance is in parameter units. The normalized median shift is (CS median − baseline median) / √(s²_baseline + s²_CS); this differs from the earlier normalization by the baseline 90% interval width. Width ratios use central 68% intervals.

Prediction diagnostics reuse the exact 250 parameter vectors per chain stored with the observable plots. Walker/step indices are checked against the HDF chains. Loaded emulator state is evaluated with the current `emulator.py` class methods, without retraining or using serialized Python method bytecode. Source hashes, package versions and checks against saved observable quantiles are recorded. These sample-based quantities have Monte Carlo uncertainty and are descriptive, not hypothesis tests.

### Parameter sensitivity

| Parameter | JS (bits) | Wasserstein | Normalized Δ median | CS/baseline 68% width |
| --- | --- | --- | --- | --- |
| BG | 0.072659 | 4.3393 | 0.63428 | 1.2893 |
| remnant_energy_loss_fraction | 0.054372 | 0.072625 | -0.19298 | 1.7228 |
| shear_viscosity_3_eta_over_s_at_kink | 0.034158 | 0.019893 | -0.37931 | 0.94349 |
| string_source_sigma_x | 0.032262 | 0.046195 | 0.41444 | 1.0984 |
| shadowing_factor | 0.024978 | 0.03844 | 0.1983 | 1.3287 |
| ylossParam4At2 | 0.023073 | 0.16787 | 0.27391 | 1.0094 |
| eps_switch | 0.021358 | 0.015587 | 0.26471 | 0.80217 |
| string_source_sigma_eta | 0.020278 | 0.06381 | -0.30213 | 0.94059 |
| ylossParam4At4 | 0.018068 | 0.08913 | 0.18276 | 1.0003 |
| bulk_viscosity_3_zeta_over_s_max | 0.015439 | 0.010704 | -0.1755 | 0.88427 |
| bulk_viscosity_3_zeta_over_s_T_peak_in_GeV | 0.012269 | 0.011174 | 0.26144 | 1.0378 |
| ylossParam4At6 | 0.0064636 | 0.022087 | 0.011818 | 0.93546 |
| tau_form_mean | 0.0045775 | 0.018117 | -0.022912 | 1.0742 |
| ylossParam4var | 0.0042477 | 0.028718 | -0.084037 | 1.0912 |
| shear_viscosity_3_eta_over_s_T_kink_in_GeV | 0.0033208 | 0.004571 | -0.092444 | 0.94717 |
| bulk_viscosity_3_zeta_over_s_width_in_GeV | 0.0026342 | 0.0037231 | -0.12006 | 1.0077 |
| bulk_viscosity_3_zeta_over_s_lambda_asymm | 0.0017121 | 0.031049 | 0.071197 | 1.0481 |
| stringTransverseShiftFrac | 0.0015272 | 0.02295 | 0.094844 | 0.99568 |
| shear_viscosity_3_eta_over_s_high_T_slope_in_GeV | 0.0013775 | 0.064756 | -0.0658 | 1.0252 |
| shear_viscosity_3_eta_over_s_low_T_slope_in_GeV | 0.00034049 | 0.0063745 | -0.0034924 | 1.0067 |

The largest binned marginal change is `BG` (JS = 0.07266 bits). A shape change need not be captured by the median shift alone.

### Changes in parameter correlations

Pearson correlations are computed from the two sampled joint posteriors. These are posterior parameter correlations, not the experimental error correlation ρ. The ten largest absolute changes are:

| Parameter 1 | Parameter 2 | Baseline Pearson r | CS Pearson r | Δr |
| --- | --- | --- | --- | --- |
| ylossParam4var | remnant_energy_loss_fraction | 0.053963 | 0.54039 | 0.48643 |
| ylossParam4At4 | remnant_energy_loss_fraction | -0.064755 | -0.51785 | -0.45309 |
| ylossParam4At6 | remnant_energy_loss_fraction | 0.14484 | -0.28635 | -0.4312 |
| ylossParam4var | BG | -0.032317 | -0.42716 | -0.39484 |
| ylossParam4At4 | BG | -0.096309 | 0.29656 | 0.39287 |
| remnant_energy_loss_fraction | tau_form_mean | 0.11638 | 0.4795 | 0.36312 |
| remnant_energy_loss_fraction | eps_switch | -0.21992 | 0.12649 | 0.34641 |
| ylossParam4At4 | ylossParam4At6 | -0.054355 | 0.29013 | 0.34448 |
| ylossParam4var | string_source_sigma_x | 0.11153 | -0.23073 | -0.34227 |
| remnant_energy_loss_fraction | string_source_sigma_eta | 0.28074 | -0.052307 | -0.33304 |

### Experimental covariance structure

Both models retain the same diagonal variances. Baseline experimental errors are diagonal; CS experimental errors form one fully correlated block per collision system. At ρ = 1, the experimental CS matrix is rank one and cannot define an ordinary multivariate Gaussian likelihood on its own. Tiny negative eigenvalues within the reported tolerance are numerical roundoff. Singular matrices have undefined ordinary log determinant and infinite condition number; they are shown as “singular / undefined”, rather than regularized silently. Trace and norm depend on the observable coordinates and units.

| System | Model | Bins | Rank | Trace | Frobenius norm | Min eigenvalue | Condition number | Log determinant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Au-Au-200 | baseline | 791 | 791 | 4.8868 | 0.2545 | 4.4e-09 | 9.5626e+06 | -5263.2 |
| Au-Au-200 | cs | 791 | 1 | 4.8868 | 4.8868 | -2.2069e-15 | singular / undefined | singular / undefined |
| d-Au-200 | baseline | 126 | 126 | 1.5564 | 0.16901 | 3.2e-06 | 13972 | -711.99 |
| d-Au-200 | cs | 126 | 1 | 1.5564 | 1.5564 | -3.4196e-16 | singular / undefined | singular / undefined |

| System | ‖CS − baseline‖F / ‖baseline‖F | Max absolute difference | Trace difference |
| --- | --- | --- | --- |
| Au-Au-200 | 19.176 | 0.041833 | 0 |
| d-Au-200 | 9.1543 | 0.037381 | 0 |

### Posterior variation versus experimental uncertainty

For each observable bin, σ_post is the sample standard deviation of emulator **means** over the 250 posterior parameter vectors, in physical observable coordinates. The ratio σ_post / σ_exp compares parameter-induced variation with the saved experimental standard deviation. It excludes emulator-noise and experimental-noise draws. CV ratio = |σ_post / posterior median| / |σ_exp / experiment|. Bins are summarized with equal weight; a ratio below one is not evidence of a better fit or calibrated coverage.

| System | Posterior | Mean σ ratio | Median σ ratio | Min | Max | Fraction below 1 | Median CV ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Au-Au-200 | baseline | 1.1584 | 0.54207 | 0.051765 | 25.785 | 0.82427 | 0.5808 |
| Au-Au-200 | cs | 1.2087 | 0.56456 | 0.043973 | 27.004 | 0.79646 | 0.59529 |
| d-Au-200 | baseline | 0.74171 | 0.67869 | 0.19625 | 2.4448 | 0.86508 | 0.61984 |
| d-Au-200 | cs | 0.74033 | 0.68002 | 0.17704 | 2.4408 | 0.84921 | 0.6251 |

Per-observable breakdown:

| System | Observable | Baseline median σ ratio | CS median σ ratio |
| --- | --- | --- | --- |
| Au-Au-200 | dNdeta_eta_cen_00_03_PHOB | 0.50553 | 0.51066 |
| Au-Au-200 | dNdeta_eta_cen_03_06_PHOB | 0.46909 | 0.5016 |
| Au-Au-200 | dNdeta_eta_cen_06_10_PHOB | 0.46553 | 0.49086 |
| Au-Au-200 | dNdeta_eta_cen_10_15_PHOB | 0.46431 | 0.49323 |
| Au-Au-200 | dNdeta_eta_cen_15_20_PHOB | 0.47195 | 0.50059 |
| Au-Au-200 | dNdeta_eta_cen_20_25_PHOB | 0.50345 | 0.53272 |
| Au-Au-200 | dNdeta_eta_cen_25_30_PHOB | 0.50756 | 0.53452 |
| Au-Au-200 | dNdeta_eta_cen_30_35_PHOB | 0.53938 | 0.56454 |
| Au-Au-200 | dNdeta_eta_cen_35_40_PHOB | 0.56804 | 0.59086 |
| Au-Au-200 | dNdeta_eta_cen_40_45_PHOB | 0.58821 | 0.61032 |
| Au-Au-200 | dNdeta_eta_cen_45_50_PHOB | 0.66353 | 0.68197 |
| Au-Au-200 | v22_eta_cen_20_70_STAR | 0.3062 | 0.32498 |
| Au-Au-200 | v22_eta_cen_03_15_PHOB | 0.16517 | 0.17602 |
| Au-Au-200 | v22_eta_cen_15_25_PHOB | 0.13175 | 0.12895 |
| Au-Au-200 | v22_eta_cen_25_50_PHOB | 0.18319 | 0.1863 |
| Au-Au-200 | v22_pt_cen_00_10_PHEN | 0.65507 | 0.70896 |
| Au-Au-200 | v22_pt_cen_10_20_PHEN | 0.81985 | 0.91614 |
| Au-Au-200 | v22_pt_cen_20_30_PHEN | 0.98967 | 1.071 |
| Au-Au-200 | v22_pt_cen_30_40_PHEN | 0.9654 | 1.0236 |
| Au-Au-200 | v22_pt_cen_40_50_PHEN | 1.0014 | 1.0539 |
| Au-Au-200 | v22_pt_cen_50_60_PHEN | 0.79107 | 0.82212 |
| Au-Au-200 | v22_pt_cen_00_05_STAR | 3.6107 | 3.7111 |
| Au-Au-200 | v22_pt_cen_05_10_STAR | 4.1931 | 4.692 |
| Au-Au-200 | v22_pt_cen_10_20_STAR | 7.4498 | 8.2211 |
| Au-Au-200 | v22_pt_cen_20_30_STAR | 9.2167 | 9.8107 |
| Au-Au-200 | v22_pt_cen_30_40_STAR | 9.1603 | 9.6389 |
| Au-Au-200 | v22_pt_cen_40_50_STAR | 8.0448 | 8.4093 |
| Au-Au-200 | v22_pt_cen_50_60_STAR | 5.9807 | 6.1534 |
| Au-Au-200 | v22_int_STAR | 21.261 | 21.694 |
| Au-Au-200 | v32_int_STAR | 1.5844 | 1.6013 |
| Au-Au-200 | meanpT_pi_STAR | 0.62879 | 0.63294 |
| Au-Au-200 | meanpT_k_STAR | 0.46378 | 0.46957 |
| Au-Au-200 | meanpT_k_PHEN | 0.29476 | 0.29859 |
| Au-Au-200 | meanpT_p_PHEN | 0.34265 | 0.34867 |
| d-Au-200 | dNdeta_eta_cen_00_05_PHEN | 0.89003 | 0.92454 |
| d-Au-200 | dNdeta_eta_cen_05_10_PHEN | 0.53942 | 0.56636 |
| d-Au-200 | dNdeta_eta_cen_10_20_PHEN | 0.57724 | 0.58531 |
| d-Au-200 | v22_eta_cen_00_05_PHEN | 0.85506 | 0.77882 |
| d-Au-200 | v22_pt_cen_00_05_PHEN | 2.3061 | 2.2712 |
| d-Au-200 | v22_pt_cen_00_10_STAR | 0.57509 | 0.59668 |

### Residual and Gaussian likelihood terms

The residual is experiment minus the coordinatewise median emulator prediction in **likelihood coordinates** (log multiplicity where applicable). C_total = C_exp + average posterior emulator covariance. This matches the summary-level construction in the source analysis: the coordinatewise median is not necessarily the prediction at a single physical parameter vector. χ²/bin is a descriptive scaling, not reduced χ² with fitted degrees of freedom. The Gaussian log likelihood includes −½ log|C_total| and the normalizing constant; it excludes priors and is not Bayesian evidence.

| System | Posterior/model | χ² | χ²/bin | Log determinant | Gaussian log likelihood |
| --- | --- | --- | --- | --- | --- |
| Au-Au-200 | baseline | 665.42 | 0.84124 | -5077.9 | 1479.4 |
| Au-Au-200 | cs | 17324 | 21.902 | -8788.2 | -4994.9 |
| d-Au-200 | baseline | 16.461 | 0.13064 | -686.75 | 219.36 |
| d-Au-200 | cs | 551.39 | 4.3761 | -1250.9 | 233.95 |

The raw table `cross_covariance_likelihoods.csv` additionally evaluates **both** experimental covariance choices at each of the same posterior draws, using that draw's emulator covariance. This isolates the covariance change at fixed parameters; it is not a reweighted posterior or a model-selection calculation. `covariance_metrics.csv` also includes mean emulator covariance and total covariance metrics. `posterior_predictive_per_bin.csv` provides residuals, marginally standardized residuals, emulator/experimental variances and uncertainty ratios. Marginally standardized residuals are not whitened residuals when the covariance is correlated.

### Likelihood comparison at the same posterior draws

Each source uses the same 250 parameter vectors under both covariance evaluations. The log-likelihood interval below is the central 68% spread across those draws; it is not a confidence interval or evidence ratio.

| System | Posterior source | Evaluated C_exp | Median χ² | LogL q16 | Median logL | LogL q84 |
| --- | --- | --- | --- | --- | --- | --- |
| Au-Au-200 | baseline | baseline | 669.85 | 1474.8 | 1477.1 | 1479 |
| Au-Au-200 | baseline | cs | 17218 | -4943.9 | -4941.7 | -4939.8 |
| Au-Au-200 | cs | baseline | 671.6 | 1473.4 | 1476.2 | 1478 |
| Au-Au-200 | cs | cs | 17219 | -4944.7 | -4942.5 | -4940.7 |
| d-Au-200 | baseline | baseline | 19.551 | 215.74 | 217.91 | 219.27 |
| d-Au-200 | baseline | cs | 568.27 | 223.04 | 225.93 | 227.67 |
| d-Au-200 | cs | baseline | 19.321 | 215.9 | 217.85 | 219.1 |
| d-Au-200 | cs | cs | 564.95 | 225.45 | 227.27 | 229.02 |

### Total covariance and emulator contribution

The variance ratio uses the diagonal of the posterior-averaged emulator covariance divided by the experimental diagonal variance, in likelihood coordinates. It measures variance magnitude, not off-diagonal structure.

| System | Posterior/model | Total rank | Bins | Total condition number | Median emulator/experimental variance |
| --- | --- | --- | --- | --- | --- |
| Au-Au-200 | baseline | 791 | 791 | 2.6561e+07 | 0.66943 |
| Au-Au-200 | cs | 791 | 791 | 7.9025e+08 | 0.68384 |
| d-Au-200 | baseline | 126 | 126 | 1.3498e+05 | 0.62048 |
| d-Au-200 | cs | 126 | 126 | 2.904e+07 | 0.63471 |

### Temporal interval checks

`time_window_intervals.csv` lists the median and central 60% / 90% sample intervals in ten non-overlapping windows for each chain. These describe temporal changes in the sampled distributions and do not prove convergence. No τ, ESS or walker-based R-hat is added to this report.



The table reports the range of window medians divided by the median window 90% interval width. It measures temporal movement relative to the sampled spread; it has no convergence pass/fail threshold.

| Parameter | Baseline temporal ratio | CS temporal ratio |
| --- | --- | --- |
| ylossParam4At2 | 0.033436 | 0.060856 |
| ylossParam4At4 | 0.039288 | 0.073868 |
| ylossParam4At6 | 0.07211 | 0.040528 |
| ylossParam4var | 0.044882 | 0.060444 |
| remnant_energy_loss_fraction | 0.047892 | 0.0928 |
| shadowing_factor | 0.066678 | 0.063603 |
| tau_form_mean | 0.10805 | 0.12137 |
| BG | 0.076767 | 0.23488 |
| string_source_sigma_x | 0.03601 | 0.074397 |
| string_source_sigma_eta | 0.057842 | 0.087344 |
| stringTransverseShiftFrac | 0.14854 | 0.03781 |
| shear_viscosity_3_eta_over_s_T_kink_in_GeV | 0.036059 | 0.040849 |
| shear_viscosity_3_eta_over_s_low_T_slope_in_GeV | 0.085155 | 0.092142 |
| shear_viscosity_3_eta_over_s_high_T_slope_in_GeV | 0.095257 | 0.097926 |
| shear_viscosity_3_eta_over_s_at_kink | 0.064078 | 0.05339 |
| bulk_viscosity_3_zeta_over_s_max | 0.0704 | 0.079954 |
| bulk_viscosity_3_zeta_over_s_T_peak_in_GeV | 0.066707 | 0.059992 |
| bulk_viscosity_3_zeta_over_s_width_in_GeV | 0.077815 | 0.085858 |
| bulk_viscosity_3_zeta_over_s_lambda_asymm | 0.068459 | 0.12094 |
| eps_switch | 0.045045 | 0.056376 |

### Validation and rerun

The emulator cache was saved with scikit-learn 1.6.1 and evaluated here with scikit-learn 1.8.0. Agreement with saved quantiles checks the predicted means; it does not independently validate historical emulator covariance values. Package versions are recorded in diagnostics.json.

Recomputed observable quantiles match the saved plot CSVs within rtol 10⁻⁶ and atol 10⁻¹⁰ in **80 of 80** run/observable comparisons. See `saved_quantile_validation.csv` for errors.

Run from the project directory:

```bash
python mcmc/scripts/diagnose_phase2_numerical.py
```

This updates the numerical tables and this notebook section without generating new graphics or running MCMC. The original Phase 2 notebook builder preserves this section on regeneration.
