# Posterior corner plot

`plot_posterior_corner.py` reads the JETSCAPE `chain` HDF5 dataset in
`(walkers, production_steps, parameters)` order without importing the emulator.
Parameter names and prior limits come from the production design CSV.

Dependencies: `python3 -m pip install numpy scipy matplotlib h5py`.
Run from this directory (or use the script's absolute path):

```sh
python3 plot_posterior_corner.py
```

Outputs: `plots/posterior_corner.pdf`, `.png`, and `.json` (quantiles and
sampling settings). The default plot includes all 20 parameters. To reproduce
the compact initial-condition subset seen in the reference:

```sh
python3 plot_posterior_corner.py --parameters 0 1 2 3 4 5 6 7 8 \
    --output plots/posterior_initial_conditions
```

Options include `--burnin 1000`, `--thin 10`, `--max-samples 200000`,
`--ci 0.90`, `--bins 45`, `--smooth 1`, and `--logo /path/to/logo.png`.
`--parameters` also accepts exact parameter names from the design CSV.
`--chain` and `--design` allow another chain and matching parameter schema.

The original sampler discards burn-in before saving production steps, so the
default additional burn-in is zero. This does not establish convergence; choose
additional discard using chain diagnostics. To bound memory, the script may
increase thinning automatically, reporting and recording the actual stride.
All quantiles and densities use that same retained sample set.

Visual conventions: empirical green step histograms; central red dashed lines
at marginal medians; green dashed bounds of the 90% equal-tailed interval;
smoothed 2D histograms with green highest-density contours containing 68% and
90% of the gridded probability; no sample scatter. Histograms use full prior
limits. Large mathematical labels appear in an external column. The upper
triangle remains white; an optional supplied logo can occupy it. No logo is
bundled. PDF density layers are rasterized, while text and lines remain vector.

Display labels follow the article reference images (including stacked eta/s and
zeta/s fractions). The default chain is `chain-idf-0_mine.hdf`; use `--chain`
for another filename. HDF5 parameter keys and numeric values are unchanged.
