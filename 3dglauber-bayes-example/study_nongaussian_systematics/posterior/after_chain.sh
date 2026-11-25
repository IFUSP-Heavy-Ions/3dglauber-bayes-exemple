#!/bin/bash
# Wait for the nuisance chain job, then build the 3-way comparison.
cd /storage/avilamrs/code-space/my-work/jetscape-emulator/3dglauber-bayes-example
JOB=$1
while condor_q $JOB -af JobStatus 2>/dev/null | grep -q .; do sleep 120; done
echo "job $JOB left the queue: $(date)"
grep -E "Traceback|writing chain" study_nongaussian_systematics/posterior/logs/nuisance.$JOB.err | tail -2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice .venv/bin/python study_nongaussian_systematics/posterior/compare_posteriors.py
echo "compare exit $? at $(date)"
