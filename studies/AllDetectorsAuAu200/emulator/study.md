# AllDetectorsAuAu200 Emulator

## Purpose
This document is the release note template for the `emulator` study under `AllDetectorsAuAu200`.
Use it to describe emulator validation, emulator equivalence checks, and the emulator artifact selected for release.

## Study Scope
- System: `AllDetectorsAuAu200`
- Study: `emulator`
- Objective: validate and release the emulator configuration used for the Au+Au / d+Au analysis
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
- Emulator implementation:
- Design transform:
- Observable-space checks:
- Equivalence diagnostics:

## Expected Outputs
- Emulator cache files
- Equivalence diagnostics
- Comparison plots
- Tables summarizing emulator agreement

## Release Checklist
- [ ] Final YAML/config set reviewed
- [ ] Chosen emulator artifact identified
- [ ] Input tables frozen
- [ ] Emulator diagnostics rerun
- [ ] Comparison plots regenerated
- [ ] Equivalence summary updated
- [ ] Known caveats documented below

## Reproducibility Notes
- Training command:
- Diagnostic command:
- Plotting command:
- Required environment variables:

## Released Artifacts
- Emulator artifact:
- Diagnostics:
- Plots:
- Tables:
- Logs:

## Validation Summary
- Reference emulator:
- hic-bayes emulator:
- Main agreement metrics:
- Remaining discrepancies:

## Known Limitations
- None documented yet.

## Change Log
- Initial `study.md` template added to standardize release notes for study results.