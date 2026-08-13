#!/usr/bin/env bash

if [[ -d /storage ]]; then
	export HIC_BAYES_RESULTS="/storage/${USER}/results/hic-bayes"
else
	echo "ERROR: /storage is not available. This HPC workflow requires the /storage mount." >&2
	exit 1
fi