# AllDetectorsAuAu200 Uncertainty Correlation

## Purpose
This document records the scope, technical configuration, observables, parameters, and release context for the `uncertainty_correlation` study under `AllDetectorsAuAu200`.
The study is designed to quantify how the posterior changes when experimental correlations are added to the likelihood, comparing a diagonal baseline against constant/equicorrelated (CS) and RBF-style correlation models.

## Study Scope
- System: `AllDetectorsAuAu200`
- Study: `uncertainty_correlation`
- Objective: add correlated experimental uncertainties to the likelihood and measure their impact on the posterior for RHIC top-energy Au+Au and d+Au observables
- Physics target: longitudinal dynamics in large and small systems, calibrated with RHIC top-energy collision data
- Correlation models studied:
	- baseline diagonal covariance
	- CS/equicorrelated covariance scans with `rho = 0.1, 0.2, ..., 0.9`
	- RBF covariance scans with `ro = 0.0, 0.1, 0.5, 0.9, 1.3, 1.7, 2.0, 4.0, 6.0, 8.0, 10.0`
- Status: active analysis and release note

## Release Metadata
- Release tag:
- Release date:
- Author:
- Branch/commit:
- Results root:
- Chains root:
- Article link: https://doi.org/10.1103/7kjw-vfgp

## Technical Configuration
- Primary simulation configs: `sims/`
- Plot configuration: `plot/`
- Batch submission scripts: `scripts/` and local `submit_*.sh`
- Data inputs:
	- `data/AllDetectorsAuAu200/Au-Au-200/experiment.csv`
	- `data/AllDetectorsAuAu200/Au-Au-200/exp_error.csv`
	- `data/AllDetectorsAuAu200/d-Au-200/experiment.csv`
	- `data/AllDetectorsAuAu200/d-Au-200/exp_error.csv`
- Model inputs:
	- `model/AllDetectorsAuAu200/Au-Au-200/design_points.csv`
	- `model/AllDetectorsAuAu200/Au-Au-200/simulation.csv`
	- `model/AllDetectorsAuAu200/d-Au-200/design_points.csv`
	- `model/AllDetectorsAuAu200/d-Au-200/simulation.csv`
- Emulator configuration:
	- hic-bayes-trained emulator (`train_in_hic_bayes: true`)
	- 10 principal components
	- 4 GP restarts
	- cached separately for Au+Au 200 and d+Au 200
	- multiplicity transformation enabled
	- monotonicity constraint for the first three parameters enabled
- Sampler backend:
	- `emcee`
	- 200 walkers
	- 175 production steps
	- 100 burn-in steps
	- thinning factor 10
	- 3 threads
	- `n_temps: 20` remains present in the shared study schema although this study runs with `emcee`
- Likelihood space:
	- observable-space likelihood with correlated experimental covariance injected directly into the likelihood evaluation
- Correlation modes released:
	- `diag` baseline
	- `equicorr` scans parameterized by `rho`
	- `rbf_correlation` scans parameterized by `ro`
- Reference run IDs:
	- release-specific run IDs should be recorded when freezing a release

## Studied Parameters
The study calibrates the same 20-parameter model for both Au+Au 200 and d+Au 200:

1. `ylossParam4At2` as $y_2$
2. `ylossParam4At4` as $y_4$
3. `ylossParam4At6` as $y_6$
4. `ylossParam4var` as $\sigma_y$
5. `remnant_energy_loss_fraction` as $f_{\rm rem}$
6. `shadowing_factor` as $S_{\rm shad}$
7. `tau_form_mean` as $\tau_{\rm form}$
8. `BG` as $BG$
9. `string_source_sigma_x` as $\sigma_x$
10. `string_source_sigma_eta` as $\sigma_\eta$
11. `stringTransverseShiftFrac` as $f_{\rm shift}$
12. `shear_viscosity_3_eta_over_s_T_kink_in_GeV` as $T_{\eta,\rm kink}$
13. `shear_viscosity_3_eta_over_s_low_T_slope_in_GeV` as $a_{\eta,\rm low}$
14. `shear_viscosity_3_eta_over_s_high_T_slope_in_GeV` as $a_{\eta,\rm high}$
15. `shear_viscosity_3_eta_over_s_at_kink` as $(\eta/s)_{\rm kink}$
16. `bulk_viscosity_3_zeta_over_s_max` as $(\zeta/s)_{\max}$
17. `bulk_viscosity_3_zeta_over_s_T_peak_in_GeV` as $T_{\zeta,\rm peak}$
18. `bulk_viscosity_3_zeta_over_s_width_in_GeV` as $w_\zeta$
19. `bulk_viscosity_3_zeta_over_s_lambda_asymm` as $\lambda_\zeta$
20. `eps_switch` as $\epsilon_{\rm sw}$

## Observables Included
The released posterior comparisons are based on the following observable groups.

### Au+Au 200
1. `dNdeta_eta_cen_00_03_PHOB`
2. `dNdeta_eta_cen_03_06_PHOB`
3. `dNdeta_eta_cen_06_10_PHOB`
4. `dNdeta_eta_cen_10_15_PHOB`
5. `dNdeta_eta_cen_15_20_PHOB`
6. `dNdeta_eta_cen_20_25_PHOB`
7. `dNdeta_eta_cen_25_30_PHOB`
8. `dNdeta_eta_cen_30_35_PHOB`
9. `dNdeta_eta_cen_35_40_PHOB`
10. `dNdeta_eta_cen_40_45_PHOB`
11. `dNdeta_eta_cen_45_50_PHOB`
12. `v22_eta_cen_20_70_STAR`
13. `v22_eta_cen_03_15_PHOB`
14. `v22_eta_cen_15_25_PHOB`
15. `v22_eta_cen_25_50_PHOB`
16. `v22_pt_cen_00_10_PHEN`
17. `v22_pt_cen_10_20_PHEN`
18. `v22_pt_cen_20_30_PHEN`
19. `v22_pt_cen_30_40_PHEN`
20. `v22_pt_cen_40_50_PHEN`
21. `v22_pt_cen_50_60_PHEN`
22. `v22_pt_cen_00_05_STAR`
23. `v22_pt_cen_05_10_STAR`
24. `v22_pt_cen_10_20_STAR`
25. `v22_pt_cen_20_30_STAR`
26. `v22_pt_cen_30_40_STAR`
27. `v22_pt_cen_40_50_STAR`
28. `v22_pt_cen_50_60_STAR`
29. `v22_int_STAR`
30. `v32_int_STAR`
31. `meanpT_pi_STAR`
32. `meanpT_k_STAR`
33. `meanpT_k_PHEN`
34. `meanpT_p_PHEN`

### d+Au 200
1. `dNdeta_eta_cen_00_05_PHEN`
2. `dNdeta_eta_cen_05_10_PHEN`
3. `dNdeta_eta_cen_10_20_PHEN`
4. `v22_eta_cen_00_05_PHEN`
5. `v22_pt_cen_00_05_PHEN`
6. `v22_pt_cen_00_10_STAR`

## Expected Outputs
- Chains for baseline and correlation variants
- Posterior comparison plots
- Sensitivity metrics and ranking tables
- Release-ready summary figures
- Paper-comparison corner plots against the published calibration when applicable

## Release Checklist
- [ ] Final YAML/config set reviewed
- [ ] Correlation scenarios selected for release
- [ ] Output directory structure checked
- [ ] Chains completed without failed jobs
- [ ] Posterior summary regenerated
- [ ] Comparison plots regenerated
- [ ] Sensitivity summary updated
- [ ] Release figures exported
- [ ] Known caveats documented below

## Reproducibility Notes
- Submission command: `studies/AllDetectorsAuAu200/uncertainty_correlation/submit_uncertainty_correlation.sh --description "..."`
- Plotting command: `studies/AllDetectorsAuAu200/uncertainty_correlation/submit_uncertainty_correlation_plots.sh --run-id run_XXX`
- Diagnostic command: run the study-specific posterior comparison and sensitivity scripts from the analysis output tree
- Required environment variables:
	- `HIC_BAYES_RESULTS`
	- `HIC_BAYES_RUN_ID` for explicit shared-run output placement
	- `PYTHON` when overriding the study virtual environment

## Released Artifacts
- Chains:
- Comparison plots:
- Sensitivity tables:
- Logs:
- Published-reference comparison plots:

## Interpretation Notes
- Baseline scenario:
	- diagonal experimental covariance used as the reference posterior for impact studies
- Equicorrelation scenarios:
	- adds constant off-diagonal structure controlled by `rho` to test the posterior response to uniform correlated systematics
- RBF/exponential scenarios:
	- adds distance-decaying correlation structure controlled by `ro` to test whether smoother experimental correlations drive different posterior shifts than the constant-correlation model
- Recommended reference posterior:
	- should be set at release time after comparing posterior stability, predictive performance, and agreement with the published calibration

## Article Reference
- Title: `Longitudinal dynamics of large and small systems from a 3D Bayesian calibration of RHIC top-energy collision data`
- Collaboration: JETSCAPE Collaboration
- Journal: `Phys. Rev. C 114, 014905 (2026)`
- Published: `2026-07-15`
- DOI: `10.1103/7kjw-vfgp`
- Link: https://doi.org/10.1103/7kjw-vfgp

## Known Limitations
- None documented yet.

## Change Log
- Initial `study.md` template added to standardize release notes for study results.
- Populated the study scope with the current observables, calibrated parameters, MCMC settings, uncertainty-correlation objective, and article reference.