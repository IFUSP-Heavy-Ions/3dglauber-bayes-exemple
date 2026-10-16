# MCMC diagnostics

Dependencies: numpy, scipy, matplotlib, h5py (same environment as corner plots).
Both scripts save to `<chain directory>/plots/convergencia` by default.

Production chain:

```sh
python3 plot_mcmc_convergence.py --chain /path/to/chain.hdf
```

Reads only `chain`, preserving `(walkers, steps, parameters)`. Outputs PDF/PNG
walker traces, time-window 90% intervals, averaged walker autocorrelation, and
integrated autocorrelation time versus chain length, plus CSV/JSON diagnostics.
`--burnin` discards saved steps before diagnostics; default 0. Decimation is
only for display: ACF uses every saved step. Tau units are saved steps, not
necessarily original sampler iterations. The 50-tau length criterion is a
heuristic, not proof of convergence. ESS is approximate and assumes
stationarity. No R-hat is computed from dependent walkers of one ensemble.
Method: https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

Saved burn-in:

```sh
python3 plot_mcmc_burnin.py --chain /path/to/chain.hdf --resampling-step 10000
```

Reads only `burnin`. Outputs PDF/PNG traces and time-window intervals, plus
JSON initial/final medians. The optional marker identifies a known resampling
boundary; walker identity is not continuous across that boundary. No stationary
tau or ESS is estimated from transient burn-in. For the local convergence_mcmc
file, 20,000 burn-in steps were saved and the sampler resamples halfway through.

Production diagnostics were generated for my_simulation and paper_simulation;
burn-in diagnostics were generated only for convergence_mcmc.

Temporal trace and window-stability plots show nested central empirical bands:
60% (20th–80th percentiles, darker green) and 90% (5th–95th percentiles,
lighter green), with the median in red. Trace bands are calculated across all
walkers at each displayed step. Window bands pool all walkers and steps in each
non-overlapping window. These describe sample coverage, not uncertainty of the
median or a proof of convergence. Both figures include a shared legend.
