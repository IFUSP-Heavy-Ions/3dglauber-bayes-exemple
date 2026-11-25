"""Experimental covariance LEARNED from cross-detector data (study_nongaussian_systematics, fatia 2a).

Only the Au-Au PHOBOS dN/deta block (11 centrality classes x 54 eta bins = 594 bins) changes; every
other bin keeps the baseline diagonal. The DIAGONAL IS UNCHANGED (each bin keeps the calibration
variance): only the correlation is learned. In the log space of the likelihood (multiplicities are
log-transformed), for PHOBOS bins i, j in classes c(i), c(j):

    Sigma_ij = a_c(i) a_c(j)                       common normalization offset (rho = 1, all classes, all eta)
             + s^2 g_c(i) g_c(j)                   centrality slope, g_c = (c - c_mean) / 100
             + delta_ij d_i                        rest of the calibration variance, uncorrelated

    a_c^2 = min_eta sigma_i^2 (class c) - u^2 - s^2 g_c^2      (so d_i >= u^2 in every bin)
    d_i   = sigma_i^2 - a_c^2 - s^2 g_c^2

What comes from data (fatia 2a, results.json):
  - u = 0.0148: class-to-class (uncorrelated) PHOBOS deviation from the three-cornered hat
    (PHENIX, PHOBOS, sPHENIX; Au+Au 200, 10 classes): 1.48 %.
  - rho_ef = 0.95-1.00 between centrality classes, covariance ~ offset + slope + small diagonal.
  - s = 0.025 per 100 % centrality: rms of the 200 GeV slopes involving PHOBOS (PHENIX-PHOBOS -0.049,
    PHOBOS-sPHENIX +0.005), divided by sqrt(2) (difference of two detectors).
Declared hypotheses (not learned):
  - the offset is flat in eta (fatia 2b, which would test it, is not done);
  - the extra forward-eta variance (shape uncertainty, R3-R5 not done) stays uncorrelated;
  - the calibration variance is kept as is (PHOBOS 90 % C.L. possibly used as 1 sigma: open point).
"""
import re

import numpy as np

PHOBOS_PATTERN = re.compile(r"^dNdeta_eta_cen_(\d+)_(\d+)_PHOB$")
DEFAULTS = dict(u=0.0148, slope=0.025)


def _phobos_blocks(slices):
    out = []
    for obs, slc in slices:
        m = PHOBOS_PATTERN.match(obs)
        if m:
            out.append((obs, slc, 0.5 * (float(m.group(1)) + float(m.group(2)))))
    return out


def learned_phobos_covariance(system, slices, std, u=DEFAULTS["u"], slope=DEFAULTS["slope"]):
    std = np.asarray(std, dtype=float)
    if std.ndim != 1 or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Standard deviations must be a finite positive vector.")
    cov = np.diag(std**2)
    blocks = _phobos_blocks(slices)
    if not blocks:
        return cov
    cmean = np.mean([c for _, _, c in blocks])
    idx = np.concatenate([np.arange(slc.start, slc.stop) for _, slc, _ in blocks])
    off = np.empty(len(idx))
    g = np.empty(len(idx))
    k = 0
    for obs, slc, c in blocks:
        n = slc.stop - slc.start
        gc = (c - cmean) / 100.0
        a2 = np.min(std[slc] ** 2) - u**2 - (slope * gc) ** 2
        if a2 <= 0:
            raise ValueError(f"{obs}: learned components exceed the calibration variance")
        off[k:k + n] = np.sqrt(a2)
        g[k:k + n] = gc
        k += n
    block = np.outer(off, off) + slope**2 * np.outer(g, g)
    d = std[idx] ** 2 - np.diag(block)
    if np.any(d < u**2 * (1 - 1e-9)):
        raise ValueError("uncorrelated part below u^2")
    cov[np.ix_(idx, idx)] = block + np.diag(d)
    if not np.allclose(np.diag(cov), std**2):
        raise AssertionError("diagonal changed")
    np.linalg.cholesky(cov)
    return cov


def summary(system, slices, std, cov):
    blocks = _phobos_blocks(slices)
    if not blocks:
        return dict(system=system, phobos_bins=0)
    idx = np.concatenate([np.arange(slc.start, slc.stop) for _, slc, _ in blocks])
    var = np.diag(cov)[idx]
    sub = cov[np.ix_(idx, idx)]
    r = sub / np.sqrt(np.outer(var, var))
    off = r[~np.eye(len(idx), dtype=bool)]
    return dict(system=system, phobos_bins=int(len(idx)), classes=len(blocks),
                corr_offdiag_mean=float(off.mean()), corr_offdiag_min=float(off.min()),
                corr_offdiag_max=float(off.max()), diag_unchanged=bool(np.allclose(np.diag(cov), np.asarray(std) ** 2)),
                min_eigenvalue=float(np.linalg.eigvalsh(cov).min()))
