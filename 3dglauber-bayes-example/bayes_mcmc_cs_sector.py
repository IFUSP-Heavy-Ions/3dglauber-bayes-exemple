#!/usr/bin/env python3
"""Run CS MCMC with correlations restricted to bins of the same observable.

Usage: python bayes_mcmc_cs_sector.py --help
No correlations between different detector/centrality/observable keys.
"""
from sector_mcmc_cli import main

if __name__ == '__main__':
    main('cs')
