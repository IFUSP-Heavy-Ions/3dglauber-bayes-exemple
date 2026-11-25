# Selection before plotting

Chain: `mcmc/01_01_baseline_reproduction/chain-idf-0.hdf`
Sampling: {'walkers': 200, 'steps': 50000, 'burnin': 0, 'thin': 50, 'samples': 200000, 'dropped_nonfinite': 0}
Representative point: coordinatewise posterior median (not MAP)

Parameters:
- shear_viscosity_3_eta_over_s_at_kink: [0.01, 0.2], index 14
- bulk_viscosity_3_zeta_over_s_max: [0.01, 0.2], index 15

Selected measurements:
- v22_pt_cen_30_40_STAR, bin 3 [0.75, 0.9]: 0.102 +/- 0.00035000000000000005; emulator output 728
- meanpT_pi_STAR, bin 5 [40.0, 50.0]: 0.395 +/- 0.02; emulator output 766

Full saved calibration, all active observables and both systems Au-Au-200 and d-Au-200; 2D marginal over other 18 parameters, NOT a new two-observable posterior.

Baseline: diagonal. The err input column is variance; std=sqrt(err), as in bayes_exp.py. No separate statistical/systematic decomposition supplied.

MAP available: False

Existing convergence diagnostics:
- shear_viscosity_3_eta_over_s_at_kink: tau=3212.6008751835866, N/tau=15.563713621021382, passes 50tau=False
- bulk_viscosity_3_zeta_over_s_max: tau=3698.9003302234814, N/tau=13.517531032521518, passes 50tau=False
