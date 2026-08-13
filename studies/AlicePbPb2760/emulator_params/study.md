# AlicePbPb2760 Emulator Parameters

## Purpose
This document is the release note template for the `emulator_params` study under `AlicePbPb2760`.
Use it to record the emulator hyperparameter scan or comparison that produced the released results.

## Study Scope
- System: `AlicePbPb2760`
- Study: `emulator_params`
- Objective: emulator hyperparameter exploration and release of the selected emulator setup
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
- Emulator family:
- PCA configuration:
- GP restart policy:
- Sampler configuration:
- Selection criterion for final emulator setup:

## Expected Outputs
- Emulator scan outputs
- Posterior chains for selected emulator setups
- Comparison plots across emulator settings
- Tables summarizing emulator performance

## Release Checklist
- [ ] Final YAML/config set reviewed
- [ ] Selected emulator setup identified
- [ ] Output directory structure checked
- [ ] Chains completed without failed jobs
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
- Selected emulator configuration:
- Chains:
- Plots:
- Tables:
- Logs:

## Recommendation
- Preferred emulator setup:
- Reason for selection:

## Known Limitations
- None documented yet.

## Change Log
- Initial `study.md` template added to standardize release notes for study results.