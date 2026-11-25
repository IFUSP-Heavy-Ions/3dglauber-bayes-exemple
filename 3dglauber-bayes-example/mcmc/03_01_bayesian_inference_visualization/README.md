# Bayesian inference visualization

Three panels, 16:9: two conditional predictions of the existing trained GP and
the two-parameter marginal of the **full baseline calibration**. No GP fitting,
optimization, or MCMC is performed.

## Reproduce

Use Python 3.9 and the dependencies in `requirements.txt`. From this directory:

```sh
python3 -m venv /path/to/local/hic-figure-venv
/path/to/local/hic-figure-venv/bin/python -m pip install -r requirements.txt
/path/to/local/hic-figure-venv/bin/python plot_inference.py --prepare-only
# Inspect outputs/selection_report.md before plotting.
/path/to/local/hic-figure-venv/bin/python plot_inference.py --plot-only
```

With dependencies already available, `python plot_inference.py` runs both phases.
The script is independent of the shell working directory but requires this
project's original data and utilities. `HIC_PYTHON_PACKAGES` optionally selects an
existing local package directory. Do not use the project's Linux virtual
environment on macOS. A local environment avoids slow package installation on
the mounted HPC filesystem.

`--grid` controls the surface resolution (default 45 per axis); `--batch` bounds
the memory used by native GP covariance prediction (default 12). Optional
`--burnin` and `--thin` require a new preparation. The existing sample loader may
increase thinning to retain at most approximately 200,000 samples.

## Inputs and interpretation

- Parameters: viscosity at the shear kink and maximum bulk viscosity, each with
  configured prior `[0.01, 0.20]`, raw chain columns 14 and 15 (zero-based).
- Conditional slice: other 18 parameters fixed at their coordinatewise posterior
  medians. No reliably identified saved MAP is available.
- STAR elliptic flow: 30-40% centrality, `0.75 < pT < 0.90 GeV/c`,
  `v2{2} = 0.102 +/- 0.00035`, output 728.
- STAR pion mean transverse momentum: 40-50% centrality,
  `0.395 +/- 0.020 GeV/c`, output 766.
- Chain: `../01_01_baseline_reproduction/chain-idf-0.hdf`, the reorganized
  `my_simulation` baseline, 200 walkers and 50,000 saved production steps.
  Existing baseline processing uses zero **additional production discard** and
  thinning 50, retaining 200,000 samples. The sampler source resets after burn-in;
  this baseline file contains only `chain`, not a recoverable burn-in history.
  No new claim about equilibration is made.
- Both systems (Au-Au and d-Au at 200 GeV) and all active calibration observables
  contribute to the saved posterior. It is **not a two-observable calibration**.
- Observable bins are selected automatically for complementary local gradients,
  weighted by experimental plus GP uncertainty, and conditional-grid coverage.
  This illustrative selection is not a discovery/significance test.
- The native `Emulator.predict()` retains input transformations and PCA/scaler
  reconstruction. Its means are checked against 34 saved observable groups.
  Source definitions are loaded without module-level training side effects.
- Experimental `err` columns contain **variance**, not standard deviation.
  The baseline `compute_cov()` is reused and verified to be diagonal.
- Green regions satisfy
  `abs(GP mean - measurement) <= sqrt(experimental variance + GP variance)`.
  These are pointwise one-sigma compatibility diagnostics, not a joint
  likelihood, confidence region, or posterior. The GP mesh includes native
  predictive covariance and PCA truncation variance.
- The 68%/95% highest-density contours use a normalized 100x100 histogram,
  Gaussian-smoothed by 1.35 bins with reflecting boundaries. They estimate
  the saved finite-chain distribution. The saved diagnostics give only about
  15.6 and 13.5 autocorrelation times for the two parameters; **neither passes
  the 50-tau criterion**. Thinning does not establish independence or convergence.

## Files

- `inputs/`: posterior snapshot and copies of small scientific/configuration
  inputs and helper sources; original chain and emulator remain read-only.
- `outputs/selection_report.md`, `selection.json`: selection printed before
  prediction/plotting, with indices, values, sampling settings and provenance.
- `outputs/observable_sensitivity_scan.json`: candidate-bin sensitivity scan.
- `outputs/prediction_grid.npz`: physical parameter grid, conditional inputs,
  GP means, variances and cross-observable covariance.
- `outputs/posterior_density.npz`: density and bin edges used for contours.
- `outputs/bayesian_inference_three_panel.pdf`: 16x9 inch presentation figure.
- `outputs/bayesian_inference_three_panel.png`: 4800x2700 pixels, 300 dpi.
- `outputs/manifest.json`: source hashes, versions, limitations, contour mass
  checks and output hashes. Replotting rejects changed snapshotted sources.
- `logs/`: execution logs and plotting cache.

For Beamer, include the PDF at `width=\textwidth`. A slide containing this
figure should retain its conditional/marginal distinction and finite-chain
warning.
