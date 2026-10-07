#!/usr/bin/env bash

set -e

echo "===== A: OpenBLAS parallel — 12 threads ====="

BENCH_MODE=serial \
OPENBLAS_NUM_THREADS=12 \
python benchmark_parallelism.py


echo
echo "===== B: likelihoods parallel — 12 processes ====="

BENCH_MODE=pool \
OPENBLAS_NUM_THREADS=1 \
NPROCESSES=12 \
python benchmark_parallelism.py