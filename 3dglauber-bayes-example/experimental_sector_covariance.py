"""Experimental covariance blocks for individual observable keys.

RBF convention: R_ij = exp(-((x_i-x_j)/length)**2), without a factor 1/2.
Lengths refer to the independent bin coordinate, never the observable's value.
"""
import hashlib
import json
import numpy as np
from experimental_cs_covariance import compound_symmetry_covariance

AXIS_UNITS = {'pt': 'GeV/c', 'eta': 'dimensionless pseudorapidity',
              'centrality': 'percentage points (0 to 100)'}


def observable_axis(observable):
    if '_pt_' in observable:
        return 'pt'
    if '_eta_' in observable:
        return 'eta'
    if '_int_' in observable or observable.startswith('meanpT_'):
        return 'centrality'
    raise ValueError(f'Unknown bin coordinate for {observable!r}; define its axis explicitly.')


def bin_centers(edges, expected_size=None):
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.isfinite(edges).all():
        raise ValueError('Expected finite physical bin edges shaped (n_bins, 2).')
    if np.any(edges[:, 1] <= edges[:, 0]):
        raise ValueError('Bin upper edges must exceed lower edges.')
    if expected_size is not None and len(edges) != expected_size:
        raise ValueError('Physical bin count does not match observable data.')
    centers = edges.mean(axis=1)
    if len(np.unique(centers)) != len(centers):
        raise ValueError('Duplicate bin centers require an explicit modeling decision.')
    return centers


def rapidity_region_labels(system, observable, edges, regions=None):
    """Use explicit eta boundaries; do not infer detector sectors from sparse bins.

    regions = {system: {observable: {"boundaries": [-1, 1],
                                   "labels": ["backward", "mid", "forward"]}}}
    A bin may touch a boundary, but may not straddle it. Non-eta observables
    remain one sector unless a future study defines another coordinate policy.
    """
    centers = bin_centers(edges)
    spec = (regions or {}).get(system, {}).get(observable)
    if spec is None:
        return np.repeat('all', len(centers))
    if observable_axis(observable) != 'eta':
        raise ValueError(f'Rapidity regions apply only to eta observables: {system}/{observable}.')
    boundaries = np.asarray(spec['boundaries'], dtype=float)
    labels = spec['labels']
    if (boundaries.ndim != 1 or not np.isfinite(boundaries).all()
            or np.any(np.diff(boundaries) <= 0)):
        raise ValueError('Rapidity boundaries must be finite and strictly increasing.')
    if len(labels) != len(boundaries)+1 or not all(isinstance(v,str) and v for v in labels):
        raise ValueError('Use one nonempty label per rapidity interval.')
    edges = np.asarray(edges, dtype=float)
    for boundary in boundaries:
        crosses = (edges[:,0] < boundary-1e-10) & (edges[:,1] > boundary+1e-10)
        if crosses.any():
            raise ValueError(f'Rapidity boundary {boundary} cuts through a bin: {system}/{observable}.')
    return np.asarray(labels)[np.searchsorted(boundaries, centers, side='right')]


def validate_regions(regions, bins):
    if not isinstance(regions,dict):
        raise ValueError('Rapidity regions must be an object keyed by system and observable.')
    for system, mapping in regions.items():
        if system not in bins or not isinstance(mapping,dict):
            raise ValueError(f'Unknown system or invalid rapidity regions: {system}.')
        for observable, spec in mapping.items():
            if observable not in bins[system]:
                raise ValueError(f'Unknown rapidity observable: {system}/{observable}.')
            if not isinstance(spec,dict) or set(spec) != {'boundaries','labels'}:
                raise ValueError('Each rapidity specification needs boundaries and labels.')
            rapidity_region_labels(system,observable,bins[system][observable],regions)


def likelihood_std(observable, mean, std, transform_multiplicities=True):
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError('Experimental means and standard deviations must be matching vectors.')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError('Experimental values must be finite; errors must be nonnegative.')
    if transform_multiplicities and ('dN' in observable or 'dET' in observable):
        if np.any(mean <= 0):
            raise ValueError('Log-multiplicity means must be positive.')
        return np.sqrt(np.log1p((std / mean)**2))
    return std.copy()


def resolve_length(system, observable, lengths, overrides):
    axis = observable_axis(observable)
    value = overrides.get(system, {}).get(observable, lengths.get(axis))
    if value is None:
        raise ValueError(f'Missing RBF length for {system}/{observable}: supply --rbf-length-{axis} '
                         f'or a per-observable override ({AXIS_UNITS[axis]}).')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'RBF length must be finite and positive: {system}/{observable}.')
    return value


def validate_overrides(overrides, bins):
    if not isinstance(overrides, dict):
        raise ValueError('RBF length overrides must be an object keyed by system and observable.')
    for system, mapping in overrides.items():
        if system not in bins or not isinstance(mapping, dict):
            raise ValueError(f'Unknown system or invalid overrides for {system!r}.')
        for observable, value in mapping.items():
            if observable not in bins[system]:
                raise ValueError(f'Unknown observable override: {system}/{observable}.')
            if not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError('Every RBF length override must be finite and positive.')


def sector_covariance(system, slices, std, bins, model, rho=1., lengths=None, overrides=None, regions=None):
    """Zero cross-observable and cross-region covariance, preserving bin order."""
    std = np.asarray(std, dtype=float)
    if std.ndim != 1 or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError('Standard deviations must be a finite nonnegative vector.')
    if model not in ('cs', 'rbf'):
        raise ValueError('Sector covariance model must be cs or rbf.')
    covariance = np.zeros((len(std), len(std)))
    covered = np.zeros(len(std), dtype=bool)
    for observable, slc in slices:
        indices = np.arange(len(std))[slc]
        if not len(indices) or covered[indices].any():
            raise ValueError('Observable slices must be nonempty and disjoint.')
        covered[indices] = True
        centers = bin_centers(bins[system][observable], len(indices))
        labels = rapidity_region_labels(system,observable,bins[system][observable],regions)
        # Construct each sub-block independently, preserving PSD and enforcing
        # negative-rho limits on the actual region sizes rather than the full obs.
        for label in np.unique(labels):
            selected = np.flatnonzero(labels == label)
            target = indices[selected]
            sig = std[target]
            if model == 'cs':
                block = compound_symmetry_covariance(sig, rho=rho)
            else:
                length = resolve_length(system, observable, lengths or {}, overrides or {})
                x = centers[selected]
                with np.errstate(over='ignore'):
                    distance = (x[:, None] - x[None, :]) / length
                    correlation = np.exp(-distance**2)
                block = np.outer(sig, sig) * correlation
            covariance[np.ix_(target,target)] = block
    if not covered.all():
        raise ValueError('Observable slices must cover every experimental bin.')
    return covariance


def covariance_signature(matrices, slices, bins):
    """Fingerprint exact likelihood blocks and physical bin definitions for resume."""
    digest = hashlib.sha256()
    for system in sorted(matrices):
        digest.update(system.encode())
        digest.update(np.asarray(matrices[system], dtype='<f8').tobytes())
        for obs, _ in slices[system]:
            digest.update(obs.encode())
            digest.update(np.asarray(bins[system][obs], dtype='<f8').tobytes())
    return digest.hexdigest()
