"""Global RBF experimental correlation in a common physical space (pT, eta, centrality).

Method (advisor's whiteboard, 24/11/2026; img/RBF_GLOBAL_METHOD.md in the journal):
every bin of every observable of one collision system is a point x = (pT, eta, cent) and

    R_ij = exp(-((pT_i-pT_j)/L_pt)^2) * exp(-((eta_i-eta_j)/L_eta)^2) * exp(-((c_i-c_j)/L_cent)^2)

with ONE length per axis in physical units ("common range"), so each ratio is dimensionless.
Same convention as experimental_sector_covariance (no factor 1/2).

Coordinates of a bin:
  - the differential axis of the observable: bin center (bins_and_cuts.obs_cent_list);
  - centrality: the centrality class center from the observable key (cen_AA_BB), or the bin
    center for observables differential in centrality;
  - an integrated axis (eta or pT): the MIDPOINT of the measurement's acceptance window,
    from the acceptance table (decision of 24/11/2026; the spectrum mode was considered and dropped).
Cross-observable factor: neutral (alpha = 1 for every pair inside a system).
Different systems (Au-Au, d-Au) are never correlated (one covariance per system).
A product of Gaussian kernels on fixed coordinates is positive semidefinite.
"""
import json
import re
from pathlib import Path

import numpy as np

from experimental_sector_covariance import bin_centers, observable_axis

AXES = ("pt", "eta", "centrality")
AXIS_UNITS = {"pt": "GeV/c", "eta": "dimensionless pseudorapidity", "centrality": "percentage points"}
DEFAULT_ACCEPTANCE = Path(__file__).with_name("global_rbf_acceptance.json")


def centrality_center(observable):
    m = re.search(r"cen_(\d+)_(\d+)", observable)
    if m is None:
        raise ValueError(f"No centrality class in {observable!r}")
    return 0.5 * (float(m.group(1)) + float(m.group(2)))


def load_acceptance(path=DEFAULT_ACCEPTANCE):
    table = json.loads(Path(path).read_text())
    table.pop("_meta", None)
    return table


def representative(acceptance, system, observable, axis):
    """Midpoint of the finite acceptance window [lo, hi] of an integrated axis."""
    try:
        lo, hi = acceptance[system][observable][axis]["window"]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Missing {axis} window for {system}/{observable} in the acceptance table")
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise ValueError(f"{axis} window must be finite with lo <= hi: {system}/{observable}")
    return 0.5 * (lo + hi)


def bin_coordinates(system, observable, edges, acceptance):
    """(n_bins, 3) array of (pT, eta, centrality) for one observable."""
    axis = observable_axis(observable)
    centers = bin_centers(edges)
    n = len(centers)
    coords = np.empty((n, 3))
    for k, name in enumerate(AXES):
        if name == axis:
            coords[:, k] = centers
        elif name == "centrality":
            coords[:, k] = centrality_center(observable)
        else:
            coords[:, k] = representative(acceptance, system, observable, name)
    return coords


def system_coordinates(system, slices, bins, acceptance):
    blocks = [bin_coordinates(system, obs, bins[system][obs], acceptance) for obs, _ in slices]
    return np.concatenate(blocks)


def global_rbf_covariance(system, slices, std, bins, lengths, acceptance):
    std = np.asarray(std, dtype=float)
    if std.ndim != 1 or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError("Standard deviations must be a finite nonnegative vector.")
    for name in AXES:
        value = lengths.get(name)
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Global RBF needs a finite positive length for axis {name} ({AXIS_UNITS[name]}).")
    x = system_coordinates(system, slices, bins, acceptance)
    if len(x) != len(std):
        raise ValueError("Coordinates do not cover every experimental bin.")
    scaled = x / np.array([lengths[name] for name in AXES])[None, :]
    d2 = ((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(-1)
    return np.outer(std, std) * np.exp(-d2)
