# Posterior corner plot

Requires numpy, scipy, matplotlib and h5py. The script locates the production
parameter design relative to its own location, so it works from any working
directory. Supply the HDF5 file explicitly:

```sh
python3 plot_posterior_corner.py --chain /path/to/chain.hdf5
```

Default outputs: `/path/to/plots/posterior_corner.pdf`, `.png`, and `.json`.
The plot includes all parameters unless `--parameters` is supplied:

```sh
python3 plot_posterior_corner.py --chain /path/to/chain.hdf5 --parameters 0 1 2 3 4 5 6 7 8
```

Use `--output /path/to/custom_stem` to override the destination or filename.
Other settings (`--burnin`, `--thin`, `--ci`, `--bins`, `--smooth`, `--logo`,
`--max-samples`, `--design`) and the existing visual style are preserved.
