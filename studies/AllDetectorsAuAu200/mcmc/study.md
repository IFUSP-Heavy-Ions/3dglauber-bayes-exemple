# AllDetectorsAuAu200 MCMC

## Purpose
This document is the release note template for the `mcmc` study under `AllDetectorsAuAu200`.
Use it to record sampler behavior, production settings, and the run configuration selected for released posterior chains.

## Study Scope
- System: `AllDetectorsAuAu200`
- Study: `mcmc`
- Objective: evaluate and release the MCMC configuration for the Au+Au / d+Au analysis
- Status: draft until the release fields below are completed

## Release Metadata
- Release tag:
- Release date:
- Author:
- Branch/commit:
- Results root:
- Chains root:

## Technical Configuration
- Primary configs:
- Plot configuration:
- Batch submission scripts:
- Data inputs:
- Model inputs:
- Emulator configuration:
- Sampler backend:
- Walkers:
- Steps:
- Temperatures:
- Burn-in / thinning:
- Diagnostics reviewed:

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