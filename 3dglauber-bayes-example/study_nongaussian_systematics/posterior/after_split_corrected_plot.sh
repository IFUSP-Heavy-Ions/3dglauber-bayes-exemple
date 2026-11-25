#!/bin/bash
# Wait for the 4-way comparison of the detector-split chain, then draw the corrected-data plot.
cd /storage/avilamrs/code-space/my-work/jetscape-emulator/3dglauber-bayes-example
LOG=study_nongaussian_systematics/posterior/logs/after_chain_split.$1.log
until grep -q "compare exit" $LOG 2>/dev/null; do sleep 120; done
.venv/bin/python study_nongaussian_systematics/posterior/plot_star_corrected.py \
    --dir mcmc/02_11_nongaussian_nuisance_detector_split \
    --sources study_nongaussian_systematics/data/detector_split/inferred_sources.npz \
    --label "Nuisance, TPC/FTPC normalizations" --previous mcmc/02_10_nongaussian_nuisance
echo "corrected plot exit $? at $(date)"
