# AlicePbPb2760 MCMC Parameters

## Purpose
This document is the release note template for the `mcmc_params` study under `AlicePbPb2760`.
Use it to record the sampler configuration scan and justify the MCMC setup selected for production runs.

## Study Scope
- System: `AlicePbPb2760`
- Study: `mcmc_params`
- Objective: evaluate MCMC settings and release the chosen production configuration
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
- Sampler backend (`emcee` or `ptemcee`):
- Walkers:
- Steps:
- Temperatures:
- Burn-in / thinning:
- Acceptance/mixing diagnostics used for selection:

## Expected Outputs
- Chains for each MCMC setup
- Convergence and trace diagnostics
- Posterior comparison plots across sampler settings
- Recommendation for production sampling

## Release Checklist
- [ ] Final YAML/config set reviewed
- [ ] Production sampler setup selected
- [ ] Output directory structure checked
- [ ] Diagnostics reviewed
- [ ] Posterior summary regenerated
- [ ] Comparison plots regenerated
- [ ] Final recommendation recorded below
- [ ] Known caveats documented below

## Reproducibility Notes
- Submission command:
- Plotting command:
- Post-processing command:
- Required environment variables:

## Released Artifacts
- Selected MCMC configuration:
- Chains:
- Diagnostics:
- Plots:
- Logs:

## Recommendation
- Preferred sampler setup:
- Reason for selection:

## Known Limitations
- None documented yet.

## Change Log
- Initial `study.md` template added to standardize release notes for study results.