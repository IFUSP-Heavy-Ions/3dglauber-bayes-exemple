# Experimental compound-symmetry correlation

`bayes_mcmc_cs_model.py` uses one experimental correlation block per collision
system, including all selected bins and different observables in that system.
Different collision systems remain independent. The default Pearson correlation
is `rho = 1`.

For the experimental standard deviations `sigma` in likelihood coordinates:

```
D = diag(sigma)
R = (1 - rho) I + rho 1 1^T
C_exp = D R D
C_total(theta) = C_emulator(theta) + C_exp
```

Thus `C_exp[i,i] = sigma[i]**2`, and off-diagonal entries are
`rho * sigma[i] * sigma[j]`, even between different observables. At `rho = 1`,
`C_exp = outer(sigma, sigma)`: one common standardized error shifts all bins.
This applies to the combined experimental uncertainties already used by the
script. It does not separate statistical and systematic error sources.
The existing multiplicity transformation is preserved: the relative uncertainty
is converted to `sqrt(log1p(relative_error**2))` for log-space covariance.

For a block of N > 1 bins, validity requires `-1/(N-1) <= rho <= 1`.
Although a pair can have Pearson correlation -1, a block with more than two bins
cannot have all pairs at -1. Invalid values raise an error.
At rho = 1 the experimental covariance is singular. The Gaussian likelihood
requires the total covariance to be positive definite; its Cholesky check raises
an error otherwise. No jitter is added to change the requested correlation.

Set the correlation with `--cs-rho 1` (or `--cs-rho 0` for independent errors).
The script retains its existing MCMC arguments; `--chain-path /path/new.hdf`
optionally selects the output. The default output filename includes
`-cs-system-rho-1`, keeping this experiment separate from existing chains.
HDF attributes record the model, scope, and rho; resuming a chain with missing or
incompatible covariance metadata is rejected.

Verification: `python -m unittest test_experimental_cs_covariance` checks
covariance construction, negative-correlation bounds, log-space conversion,
cross-observable correlations, total covariance and Gaussian likelihood against
SciPy, and rejection of a singular total covariance.
