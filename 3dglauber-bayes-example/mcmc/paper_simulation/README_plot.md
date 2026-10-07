# Paper chain plots

Generated with the shared `../my_simulation/plot_posterior_corner.py` script.
From the parent `mcmc` directory:

```sh
python3 my_simulation/plot_posterior_corner.py \
  --chain paper_simulation/posterior-chain-default.hdf \
  --output paper_simulation/plots/posterior_corner

python3 my_simulation/plot_posterior_corner.py \
  --chain paper_simulation/posterior-chain-default.hdf \
  --parameters 0 1 2 3 4 5 6 7 8 \
  --output paper_simulation/plots/posterior_initial_conditions
```

Same style and settings as `my_simulation`: 90% equal-tailed intervals,
45 bins, 2D smoothing of one bin, green densities, red medians, 17 pt
axis labels with right-aligned external row labels. Both runs retain
200,000 samples using automatic thinning of 20 steps, with no additional
burn-in discard. Parameter order is interpreted using the same 20-parameter
production design as `my_simulation`; the HDF5 file has no parameter-name
metadata. PDF, PNG and JSON summaries are saved in `plots`.
