"""Experimental covariance with equicorrelation (heterogeneous standard deviations)."""
import numpy as np


def compound_symmetry_covariance(std, rho=1.0, groups=None):
    """C_ij = rho*sigma_i*sigma_j within groups, C_ii=sigma_i**2.

    Different groups are independent. With groups=None, all bins share one
    block. Valid block correlation is -1/(n-1) <= rho <= 1 for n>1.
    rho=1 produces a positive semidefinite rank-one block, intentionally.
    """
    std = np.asarray(std, dtype=float)
    if std.ndim != 1 or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError('Standard deviations must be a finite nonnegative vector.')
    if not np.isfinite(rho) or not -1 <= rho <= 1:
        raise ValueError('Pearson rho must lie in [-1, 1].')
    groups = [0]*len(std) if groups is None else list(groups)
    if len(groups) != len(std):
        raise ValueError('One group label is required for each bin.')
    blocks = {}
    for i, group in enumerate(groups):
        blocks.setdefault(group, []).append(i)
    cov = np.zeros((len(std), len(std)))
    for group, indices in blocks.items():
        n = len(indices)
        lower = -1/(n-1) if n > 1 else -1
        if rho < lower:
            raise ValueError(f'rho={rho} is invalid for block {group!r} with {n} bins; minimum is {lower}.')
        sig = std[indices]
        block = rho*np.outer(sig, sig)
        np.fill_diagonal(block, sig**2)
        cov[np.ix_(indices,indices)] = block
    return cov
