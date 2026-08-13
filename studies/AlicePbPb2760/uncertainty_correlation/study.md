# AlicePbPb2760 Uncertainty Correlation

## Purpose
This document defines the release note for the `uncertainty_correlation` study under `AlicePbPb2760`.
Its goal is to track configuration, observables, calibrated parameters, and deliverables for posterior runs under different assumptions about experimental covariance.

## Study Scope
- System: `AlicePbPb2760`
- Study: `uncertainty_correlation`
- Objective: quantify the impact on the posterior when adding correlated experimental covariance terms (CS and RBF) to the likelihood
- Collision system: Pb+Pb at 2.76 TeV (`Grad_1` and `Grad_2` setups)
- Status: active release note

## Release Metadata
- Release tag:
- Release date:
- Author:
- Branch/commit:
- Results root:
- Chains root:
- Article link: 

## Technical Configuration
- Primary configs: `sims/`
- Plot configuration: `plot/`
- Batch submission scripts: `scripts/` and local `submit_*.sh`
- Data inputs:
	- `data/AlicePbPb2760/universality_PbPb2760/Grad/experiment.csv`
	- `data/AlicePbPb2760/universality_PbPb2760/Grad/exp_error.csv`
- Model inputs:
	- `data/AlicePbPb2760/universality_PbPb2760/Grad/design_points.csv`
	- `data/AlicePbPb2760/universality_PbPb2760/Grad/simulation.csv`
- Emulator configuration:
	- hic-bayes training enabled (`train_in_hic_bayes: true`)
	- PCA with 6 components
	- GP restarts: 4
	- task-local cache in `${HIC_BAYES_RESULTS}/AlicePbPb2760/uncertainty_correlation/emulador`
- Sampler backend:
	- production tasks currently use `emcee`
	- `n_walkers=200`, `n_steps=175`, `burnin=100`, `thin=10`, `n_threads=3`
	- two seeds used across Grad replicas (`seed=42` and `seed=43` in baseline tasks)
- Likelihood space:
	- observable-space likelihood with experimental covariance injected per scenario
- Correlation modes released:
	- diagonal baseline (`diag`)
	- zero experimental covariance baseline (`zeroexp`)
	- CS/equicorrelated scans: `rho = 0.1, 0.2, 0.4, 0.6, 0.8, 0.9`
	- RBF scans: `ro = 0.0, 2.0, 4.0, 6.0, 8.0, 10.0`
- Reference run IDs:
	- to be frozen at release time in this document and in `manifest.yaml`

## Studied Parameters
The same 17-dimensional parameter vector is calibrated across all scenarios:

1. `norm` as $N$[2.76 TeV]
2. `trento_p` as $p$
3. `sigma_k` as $\sigma_k$
4. `nucleon_width` as $w$ [fm]
5. `dmin3` as $d_{\rm min}$ [fm]
6. `tau_R` as $\tau_R$ [fm/$c$]
7. `alpha` as $\alpha$
8. `eta_over_s_T_kink_in_GeV` as $T_{\eta,\rm kink}$ [GeV]
9. `eta_over_s_low_T_slope_in_GeV` as $a_{\eta,\rm low}$ [GeV$^{-1}$]
10. `eta_over_s_high_T_slope_in_GeV` as $a_{\eta,\rm high}$ [GeV$^{-1}$]
11. `eta_over_s_at_kink` as $(\eta/s)_{\rm kink}$
12. `zeta_over_s_max` as $(\zeta/s)_{\rm max}$
13. `zeta_over_s_T_peak_in_GeV` as $T_{\zeta,c}$ [GeV]
14. `zeta_over_s_width_in_GeV` as $w_{\zeta}$ [GeV]
15. `zeta_over_s_lambda_asymm` as $\lambda_{\zeta}$
16. `shear_relax_time_factor` as $b_{\pi}$
17. `Tswitch` as $T_{\rm sw}$ [GeV]

## Observables
The experimental table contains 287 binned columns grouped into the following 7 observable families:

1. `UxT_cen_00_05_ALICE`
2. `UxT_cen_05_10_ALICE`
3. `UxT_cen_10_20_ALICE`
4. `UxT_cen_20_30_ALICE`
5. `UxT_cen_30_40_ALICE`
6. `UxT_cen_40_50_ALICE`
7. `UxT_cen_50_60_ALICE`

## Expected Outputs
Release deliverables for this study are:

1. Chains for each scenario and replica (`Grad_1`, `Grad_2`) in `chains/run_XXX/<task>/`:
	 - `chain.hdf5`
	 - `chain.csv` (when exported)
2. Execution logs in `logs/run_XXX/`:
	 - Condor `*.out`, `*.err`, `*.log`
3. Plots in `plots/run_XXX/` and comparison subdirectories:
	 - corner plots by parameter group
	 - multi-scenario comparison corners
4. Run registry metadata in `manifest.yaml`:
	 - run id, description, commit, and output locations
5. Posterior sensitivity artifacts (when diagnostics are executed):
	 - ranking/sensitivity tables
	 - summary markdown for correlation-impact interpretation

## Release Checklist
- [ ] Final YAML/config set reviewed
- [ ] Correlation scenarios selected for release
- [ ] Output directory structure checked
- [ ] Chains completed without failed jobs
- [ ] Posterior summary regenerated
- [ ] Comparison plots regenerated
- [ ] Sensitivity summary updated
- [ ] Deliverables list below linked to concrete run id
- [ ] Known caveats documented below

## Reproducibility Notes
- Submission command:
	- `studies/AlicePbPb2760/uncertainty_correlation/submit_uncertainty_correlation.sh --description "..."`
- Plotting command:
	- use `hic_bayes/plot.py` with `studies/AlicePbPb2760/uncertainty_correlation/plot/corner_plot_config.json`
- Diagnostic command:
	- run posterior comparison and sensitivity scripts on selected `run_XXX`
- Required environment variables:
	- `HIC_BAYES_RESULTS`
	- optional `HIC_BAYES_RUN_ID` for explicit run placement
	- optional `PYTHON` to override interpreter

## Released Artifacts
- Chains:
- Comparison plots:
- Sensitivity tables:
- Logs:
- Manifest entries used for release:

## Interpretation Notes
- Baseline scenario:
- Equicorrelation scenarios:
- RBF/exponential scenarios:
- Recommended reference posterior:

## Article Reference
- Title: 
- Collaboration: 
- Journal: 
- Published: 
- DOI: 
- Link: 

## Known Limitations
- None documented yet.

## Change Log
- Initial `study.md` template added to standardize release notes for study results.
- Populated observables, studied parameters, and deliverables for the AlicePbPb2760 uncertainty-correlation release.