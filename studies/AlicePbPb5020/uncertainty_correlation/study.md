# AlicePbPb5020 Uncertainty Correlation

## Purpose
This document is the release note template for the `uncertainty_correlation` study under `AlicePbPb5020`.
Use it to document the released posterior sets for the different experimental covariance assumptions.

## Study Scope
- System: `AlicePbPb5020`
- Study: `uncertainty_correlation`
- Objective: compare posterior sensitivity to experimental uncertainty correlation models
- Status: draft until the release fields below are completed

## Release Metadata
- Release tag:
- Release date:
- Author:
- Branch/commit:
- Results root:
- Chains root:

## Technical Configuration
- Primary configs: `sims/`
- Plot configuration: `plot/`
- Batch submission scripts: `scripts/` and local `submit_*.sh`
- Data inputs:
- Model inputs:
- Emulator configuration:
- Sampler backend:
- Likelihood space:
- Correlation modes released:
- Reference run IDs:

## Expected Outputs
- Chains for baseline and correlation variants
- Posterior comparison plots
- Sensitivity metrics and ranking tables
- Release-ready summary figures

## Release Checklist
- [ ] Final YAML/config set reviewed
- [ ] Correlation scenarios selected for release
- [ ] Output directory structure checked
- [ ] Chains completed without failed jobs
- [ ] Posterior summary regenerated
- [ ] Comparison plots regenerated
- [ ] Sensitivity summary updated
- [ ] Known caveats documented below

## Reproducibility Notes
- Submission command:
- Plotting command:
- Diagnostic command:
- Required environment variables:

## Released Artifacts
- Chains:
- Comparison plots:
- Sensitivity tables:
- Logs:

## Interpretation Notes
- Baseline scenario:
- Equicorrelation scenarios:
- RBF/exponential scenarios:
- Recommended reference posterior:

## Known Limitations
- None documented yet.

## Change Log
- Initial `study.md` template added to standardize release notes for study results.