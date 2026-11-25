#!/bin/bash
# Figures for jobs 5078 (RBF global, proposal 1) and 5079 (learned PHOBOS cov., proposal 3), 25/11/2026.
set -e
cd /storage/avilamrs/code-space/my-work/jetscape-emulator/3dglauber-bayes-example/mcmc/scripts/paper_figures
PY=/storage/avilamrs/code-space/my-work/jetscape-emulator/.venv/bin/python
O=../../04_paper_figures/out
REF=$O/ylims_ref_F05_global_cs.json
export OMP_NUM_THREADS=8
# proposal 1: RBF global
$PY fig_viscosity.py   --proposal global --outdir $O/global_rbf/02_F14-viscosity
$PY fig_corner.py      --proposal global --model rbf_global --outdir $O/global_rbf/01_F28-corner
$PY fig_shift.py       --proposal global --outdir $O/global_rbf/06_S1-shift
$PY fig_observables.py --proposal global --model rbf_global --ylims-from $REF --outdir $O/global_rbf/03_F05-observables
# proposal 3 (covinf): learned PHOBOS covariance
$PY fig_viscosity.py   --proposal covinf --outdir $O/covinf/02_F14-viscosity
$PY fig_corner.py      --proposal covinf --model learned_phobos --outdir $O/covinf/01_F28-corner
$PY fig_shift.py       --proposal covinf --outdir $O/covinf/06_S1-shift
$PY fig_expcorr.py     --proposal covinf --outdir $O/covinf/07_S2-expcov
$PY fig_observables.py --proposal covinf --model learned_phobos --ylims-from $REF --outdir $O/covinf/03_F05-observables
echo ALL DONE
