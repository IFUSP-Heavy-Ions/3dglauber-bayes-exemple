#!/usr/bin/env python3
"""Run RBF MCMC within exact observables using physical bin coordinates.

Usage: python bayes_mcmc_rbf_sector.py --help
Explicit lengths: pt in GeV/c, eta dimensionless, centrality percentage points.
"""
from sector_mcmc_cli import main

if __name__ == '__main__':
    main('rbf')
