# paper_figures: comparison figures, baseline × correlation models

Figure style and rules: `Academic-Research-Journaling/2026/novembro/2026-11-23-JETSCAPE/img/DESIGN_SYSTEM.md`.
Reference: Mankolli et al., PRC 114, 014905 (2026).

| Script | Figure | Inputs |
|---|---|---|
| `fig_corner.py` | F28-corner (Figs. 28+29) | chains |
| `fig_viscosity.py` | F14-viscosity (Fig. 14) | chains |
| `fig_observables.py` | F05-observables (Fig. 5) | chains + emulator (10 000 samples) |
| `fig_varratio.py` | F25-varratio (Fig. 25) | `mcmc/02_12_shift_vs_cov_ratio/bin_level.npz` |
| `fig_corrmatrix.py` | F26-corrmatrix (Fig. 26) | emulator + `mcmc/02_08_sector_input_correlations/*.npz` |
| `fig_shift.py` | S1-shift | chains + `bin_level.npz` |

Shared modules: `style.py` (colours, fonts, output), `common.py` (chain registry, sample policy,
provenance manifest), `emu.py` (emulator loading and cached predictions).

Sample policy: first 20 000 saved production steps, all 200 walkers, stride 50 → 80 000 samples per
model (all chains: burn-in 20 000). Samples and emulator predictions are cached in
`mcmc/04_paper_figures/cache/` (keyed by chain path, size and mtime).

Each figure writes `<stem>.pdf`, `<stem>.png` (300 dpi) and `<stem>.json` (chains, sample policy,
script hashes, plotted numbers) into `--outdir`.

## Commands (run from this directory)

```bash
IMG=".../2026-11-23-JETSCAPE/img"; P1="$IMG/1-First_sensitivity_test_global"; P2="$IMG/2-Second_sensitivity_test_sector"
python fig_viscosity.py   --proposal sector --outdir "$P2/02_F14-viscosity"
python fig_viscosity.py   --proposal global --outdir "$P1/02_F14-viscosity"
python fig_corner.py      --proposal sector --model cs_sector  --outdir "$P2/01_F28-corner"
python fig_corner.py      --proposal sector --model rbf_sector --outdir "$P2/01_F28-corner"
python fig_corner.py      --proposal global --model cs_global  --outdir "$P1/01_F28-corner"
python fig_observables.py --proposal sector --model cs_sector  --outdir "$P2/03_F05-observables"
python fig_observables.py --proposal sector --model rbf_sector --outdir "$P2/03_F05-observables"
python fig_observables.py --proposal global --model cs_global  --outdir "$P1/03_F05-observables"
python fig_varratio.py    --proposal global --outdir "$P1/04_F25-varratio"
python fig_varratio.py    --proposal sector --outdir "$P2/04_F25-varratio"
python fig_corrmatrix.py  --proposal sector --outdir "$P2/05_F26-corrmatrix"
python fig_corrmatrix.py  --proposal global --outdir "$P1/05_F26-corrmatrix"
python fig_shift.py       --proposal sector --outdir "$P2/06_S1-shift"
python fig_shift.py       --proposal global --outdir "$P1/06_S1-shift"
```

Environment used (24/11/2026): Python 3.11, numpy 2.4.6, scikit-learn 1.8.0, dill 0.4.1,
matplotlib 3.11, h5py, scipy (same versions as the 02_04/02_06 manifests).
`emu.emulator_correlation` passes `extra_std` as an array: `emulator.py` uses
`np.array(..., copy=False)`, which fails in numpy 2 with a scalar.

Missing: RBF global chain (`mcmc/02_02_global_rbf/`), method in `img/RBF_GLOBAL_METHOD.md`. When it
exists, set its path in `common.MODELS["rbf_global"]` and rerun the `global` commands plus every
`fig_observables.py` command (the shared y-limits include every available model).
