#!/bin/bash
# Wait for the detector-split nuisance chain, then build the 4-way comparison
# (baseline, CS sector, nuisance with global normalization, nuisance with TPC/FTPC normalizations).
cd /storage/avilamrs/code-space/my-work/jetscape-emulator/3dglauber-bayes-example
JOB=$1
while condor_q $JOB -af JobStatus 2>/dev/null | grep -q .; do sleep 120; done
echo "job $JOB left the queue: $(date)"
grep -E "Traceback|writing chain" study_nongaussian_systematics/posterior/logs/nuisance_split.$JOB.err | tail -2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice .venv/bin/python study_nongaussian_systematics/posterior/compare_posteriors.py \
    --out mcmc/02_11_nongaussian_nuisance_detector_split \
    --sources study_nongaussian_systematics/data/detector_split/inferred_sources.npz \
    --label "Nuisance, TPC/FTPC normalizations" \
    --previous mcmc/02_10_nongaussian_nuisance
echo "compare exit $? at $(date)"
