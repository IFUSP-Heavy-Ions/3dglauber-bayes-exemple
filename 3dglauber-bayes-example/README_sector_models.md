# Experimental correlations within each observable

Run from `3dglauber-bayes-example` with the existing project's sampler environment.
These scripts reuse the global CS sampler's likelihood, priors, parameter mapping,
emulator covariance and burn-in procedure. Only experimental covariance changes.
The original `bayes_mcmc_rbf_model.py` is not used or changed by this study.

Legacy dill emulators saved from `emulator.py` as a script resolve prediction
globals through `__main__`. Both sector launchers expose NumPy and the design
transformation settings there; the CLI supplies `_Covariance` when starting MCMC.
Help and covariance-only validation still avoid loading the trained emulators.

## Completed sector studies

The completed CS and RBF chains and their reports are in `mcmc/02_04_cs_sector`
and `mcmc/02_05_rbf_sector`, respectively. Each folder contains the original
HDF5 chain, a self-contained notebook with embedded plots, corner and marginal
PNG/PDF figures, parameter summaries, and all 40 observable predictions.

From the project root, regenerate the marginal/observable figures and notebooks:

```sh
python mcmc/scripts/build_sector_studies.py --model both
```

This reads saved chains and trained emulators only: it does not run MCMC,
retrain emulators, or rewrite model calculations. Corner figures are generated
separately with `mcmc/scripts/plot_posterior_corner.py --chain PATH --thin 50`.
The reports retain 200,000 production samples for parameter plots and use
1,000 posterior parameter vectors (seed 12345) for observable bands. These
bands show the central 68% variation of emulator means, not full predictive
uncertainty including emulator or experimental noise. The notebooks record
the covariance partition, source metadata, numerical validation and limitations.

Baseline comparisons are in `mcmc/02_06_baseline_vs_cs_sector` and
`mcmc/02_07_baseline_vs_rbf_sector`. Each includes a notebook, overlaid marginal
densities and joint 68%/90% corner contours, parameter-shift CSVs, and all
40 observable comparisons with individual model/data ratio panels. Baseline
prediction arrays are shared from the CS comparison's `baseline_predictions`
folder; the manifests record this dependency. Chains remain in their original
study folders and are opened read-only.

To regenerate both comparisons from the project root:

```sh
python mcmc/scripts/build_sector_comparisons.py --model both
```

Before launching Python, limit each NumPy/BLAS process to one compute thread.
This avoids oversubscribing the 20 MCMC worker processes on a 24-core node.
These environment variables are inherited by the worker processes:

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

The CLI defaults to 20 MCMC worker processes (`--ncpus 20`); override it if the
remote machine has a different allocation. `--nwalkers 200` is independent of
the worker count: workers evaluate walker proposals in parallel.

A sector is the **exact observable key within a collision system**. Different
centrality selections, detectors, particles or observable names are independent
experimental-error blocks, even when they belong to the same physical family.
For example, `v22_pt_cen_00_10_PHEN` and `v22_pt_cen_10_20_PHEN` are separate.
Within eta observables, the default `rapidity_regions.json` separates the central
acceptance from the combined lateral sector: backward and forward bins share one
sector. This means CS correlates the two lateral arms with each other; RBF also
allows correlation between them according to their physical eta separation.
Continuous PHOBOS multiplicity coverage remains one block. No cross-system
experimental correlations are introduced. Emulator covariance is still included
in the total likelihood and can have cross-observable terms.

## CS

Within each observable, `C_ij = sigma_i sigma_j rho` off the diagonal and
`C_ii = sigma_i^2`. Default rho is 1. At rho=1 each experimental block is
rank one, intentionally; the total likelihood also includes emulator covariance.
The admissible negative-rho bound is enforced separately for each sector.

```sh
python bayes_mcmc_cs_sector.py --check-covariance
python bayes_mcmc_cs_sector.py --cs-rho 1 \
  --nwalkers 200 --nburnsteps 20000 --ncpus 20 --status 1000 50000
```

## RBF with physical coordinates

Within one observable:

`C_ij = sigma_i sigma_j exp(-((x_i - x_j) / length)^2)`.

There is **no factor 1/2** in this convention, matching the supplied formula.
At a separation equal to `length`, the correlation is exp(-1). Each x is the
center of its bin from `bins_and_cuts.obs_cent_list`; bins are not placed on an
artificial evenly spaced grid. The exponential is dimensionless because x
and length use the same units:

| Observable bin axis | Option | Units |
| --- | --- | --- |
| Differential pT (`_pt_`) | `--rbf-length-pt` | GeV/c |
| Pseudorapidity (`_eta_`) | `--rbf-length-eta` | Dimensionless eta |
| Integrated flow (`_int_`) or meanpT versus centrality | `--rbf-length-centrality` | Percentage points, 0–100 |

For meanpT versus centrality, the length is in **centrality percentage points**,
not GeV/c: it describes the independent bin coordinate, not the value being
measured. No pT coordinate is invented for integrated observables. Fixed
centrality does not add another distance within a differential-observable sector.

All required axis lengths must be supplied explicitly, or set per observable.
The values below (0.5 GeV/c, 1 eta unit, 10 percentage points) are **illustrative
choices for a sensitivity run**, not inferred or recommended physical scales:

```sh
python bayes_mcmc_rbf_sector.py \
  --rbf-length-pt 0.5 --rbf-length-eta 1 --rbf-length-centrality 10 \
  --check-covariance

python bayes_mcmc_rbf_sector.py \
  --rbf-length-pt 0.5 --rbf-length-eta 1 --rbf-length-centrality 10 \
  --nwalkers 200 --nburnsteps 20000 --ncpus 20 --status 1000 50000
```

Optional `--rbf-lengths-json path.json` overrides the axis default for named
observables, allowing different physical lengths even on the same axis:

```json
{
  "Au-Au-200": {
    "v22_pt_cen_00_10_PHEN": 0.3,
    "meanpT_pi_STAR": 15.0
  }
}
```

Unknown systems/observable names, missing lengths, nonpositive lengths and
invalid bin layouts are rejected. No correlations are controlled by the old
`assume_corr_exp_error` flag: these entry points explicitly select the model.
Experimental diagonal variances follow the existing likelihood, including
`sqrt(log1p((experimental_std / experimental_mean)^2))` for log multiplicities.

## Output and continuation

Default CS output: `mcmc/chain-idf-0-cs-observable-rho-1.hdf`.
Default RBF output: `mcmc/chain-idf-0-rbf-observable-<signature>.hdf`.
Closure runs retain their validation-point prefix. `--chain-path` may select a
custom path, with nested parent directories created as needed. Use the same
model options and output path when continuing a chain.

Metadata records the observable scope, kernel convention, lengths, overrides,
and a SHA-256 signature of the exact experimental covariance matrices and bin
definitions. Resuming global, baseline, legacy unlabelled or differently
configured chains is rejected. Original global-CS metadata remains compatible.

`--check-covariance` reads experimental files and validates symmetry, positive
semidefiniteness and diagonal variances without loading emulators, creating an
HDF file or running the sampler. It reports every sector and selected length.
It checks experimental-data mode with one model and unchanged errors; closure
or altered-error checks are rejected explicitly. Successful covariance checks
are not a complete MCMC run or a convergence claim.

Regression checks:

```sh
python -m unittest test_experimental_cs_covariance test_experimental_sector_covariance -v
```

## Central and combined lateral rapidity sectors

Both entry points load `rapidity_regions.json` by default, and API `Chain`
uses it when `regions` is not explicitly supplied. Boundaries lie in actual
unmeasured gaps; they are not universal definitions of central or lateral rapidity.

| Measurements | Sectors (backward and forward are grouped together) |
| --- | --- |
| d–Au PHENIX dN/deta, each centrality | lateral −2.6 to −1.3 and 1.3 to 2.6; central −0.4 to 0.4 |
| d–Au PHENIX v2(eta) | central −0.4 to 0.4; lateral 1.0 to 3.0 (no backward bins) |
| Au–Au STAR v2(eta) | lateral −3.9 to −2.7 and 2.7 to 3.9; central −0.9 to 0.9 |
| Au–Au PHOBOS v2(eta), each centrality | lateral −5.06 to −2.94 and 2.94 to 5.06; central −1.79 to 1.79 |
| Au–Au PHOBOS dN/deta | Continuous −5.4 to 5.4: one block |

The PHOBOS v2 samples within each broad region remain together despite smaller
sampling gaps. Region choices are explicit and editable; gaps alone do not prove
that experimental errors are independent. This study assumes independence
between the central and combined lateral sectors. Both CS and RBF obey the same
partition.

For different definitions, pass `--rapidity-regions-json custom.json` with:

```json
{
  "d-Au-200": {
    "dNdeta_eta_cen_00_05_PHEN": {
      "boundaries": [-1.0, 1.0],
      "labels": ["lateral", "central", "lateral"]
    }
  }
}
```

Unlisted observables remain whole. A boundary cutting through a measured bin
is rejected rather than splitting the measurement. The check output lists region
labels and bin counts. Metadata records the entire region configuration and
rejects continuation with a different partition. Region-separated CS files have
`cs-observable-region-...-<signature>.hdf` names. RBF names already include the
covariance signature. An explicitly supplied empty JSON object reproduces the
older whole-observable partition, with separate metadata.
