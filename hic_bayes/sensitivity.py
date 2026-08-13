"""Sensitivity analysis: global (Sobol') and local (gradient) methods.

Global analysis
---------------
:func:`sobol_sensitivity` estimates first-order and total-effect Sobol'
indices using the Saltelli (2010) estimator, evaluated entirely through
the GP emulator.  Total cost ≈ (n_params + 2) × n_samples emulator calls,
which is negligible compared to running the actual simulator.

Local analysis
--------------
:func:`local_sensitivity` computes a numerical Jacobian (central differences)
at a given parameter point (typically the MAP estimate), giving a linearised
picture of which parameters drive which observables.
"""
from __future__ import annotations

import numpy as np


def sobol_sensitivity(
    emulator,
    pca,
    prior,
    n_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Emulator-based global Sobol' sensitivity analysis.

    Implements the Saltelli (2010) estimator for first-order (S_i) and
    total-effect (ST_i) indices.  Outputs are computed in observable space
    by applying the inverse-PCA transform to each emulator evaluation.

    Parameters
    ----------
    emulator : hic_bayes.emulator.Emulator
        A fitted Emulator instance.
    pca : hic_bayes.pca.PCAResult
        Fitted PCA used to map predictions back to observable space.
    prior : hic_bayes.prior.UniformPrior
        Defines parameter ranges for sampling.
    n_samples : int
        Base sample count N.  Total emulator evaluations = (n_params + 2) × N.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        first_order          : ndarray (n_params,)       aggregated S_i
        total_effect         : ndarray (n_params,)       aggregated ST_i
        first_order_per_obs  : ndarray (n_params, n_obs) S_i per observable
        total_effect_per_obs : ndarray (n_params, n_obs) ST_i per observable
        variance_total       : ndarray (n_obs,)          emulator output variance

    Notes
    -----
    Aggregation is variance-weighted: each observable is weighted by its
    contribution to total output variance, so observables with small
    absolute variation do not dominate the summary indices.

    Sobol' indices can be mildly negative due to finite-sample estimator
    bias; values near zero should be interpreted as negligible sensitivity.
    """
    ranges = prior.ranges      # (n_params, 2)
    n_params = ranges.shape[0]
    lo, hi = ranges[:, 0], ranges[:, 1]

    rng = np.random.default_rng(seed)
    A = lo + rng.uniform(size=(n_samples, n_params)) * (hi - lo)
    B = lo + rng.uniform(size=(n_samples, n_params)) * (hi - lo)

    def _eval(X: np.ndarray) -> np.ndarray:
        Z, _ = emulator.predict(X)
        return pca.inverse_transform(Z)  # (N, n_obs)

    f_A = _eval(A)   # (N, n_obs)
    f_B = _eval(B)

    n_obs = f_A.shape[1]
    var_total = np.var(np.vstack([f_A, f_B]), axis=0)   # (n_obs,)
    var_safe = np.where(var_total > 0, var_total, 1.0)

    S1 = np.zeros((n_params, n_obs))
    ST = np.zeros((n_params, n_obs))

    for i in range(n_params):
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        f_ABi = _eval(A_Bi)
        # Saltelli 2010, Eq. 12 & 13
        S1[i] = np.mean(f_B * (f_ABi - f_A), axis=0) / var_safe
        ST[i] = 0.5 * np.mean((f_A - f_ABi) ** 2, axis=0) / var_safe

    # Variance-weighted aggregation over observables
    total_var = var_total.sum()
    weights = var_total / total_var if total_var > 0 else np.ones(n_obs) / n_obs

    return {
        "first_order": S1 @ weights,
        "total_effect": ST @ weights,
        "first_order_per_obs": S1,
        "total_effect_per_obs": ST,
        "variance_total": var_total,
    }


def local_sensitivity(
    emulator,
    pca,
    theta: np.ndarray,
    prior,
    step_fraction: float = 0.01,
) -> dict:
    """Numerical Jacobian of emulator predictions at a parameter point.

    Uses central differences with step size = ``step_fraction × range_width``
    per parameter.

    Parameters
    ----------
    emulator : hic_bayes.emulator.Emulator
    pca : hic_bayes.pca.PCAResult
    theta : ndarray, shape (n_params,)
        Reference point (e.g. the MAP estimate).
    prior : hic_bayes.prior.UniformPrior
        Used to compute per-parameter step sizes.
    step_fraction : float
        Step as a fraction of each parameter's range width.

    Returns
    -------
    dict
        jacobian            : ndarray (n_params, n_obs)  ∂y_j / ∂θ_i
        jacobian_normalised : ndarray (n_params, n_obs)  elasticity
            Each row divided by range_width, each column divided by |y0_j|,
            giving the fractional change in observable j per unit fractional
            change of parameter i across its full range.
        theta : ndarray (n_params,)
        y0    : ndarray (n_obs,)  predictions at theta
    """
    ranges = prior.ranges
    steps = step_fraction * (ranges[:, 1] - ranges[:, 0])

    def _eval(t: np.ndarray) -> np.ndarray:
        Z, _ = emulator.predict(t.reshape(1, -1))
        return pca.inverse_transform(Z).ravel()

    y0 = _eval(theta)
    n_params = len(theta)
    n_obs = len(y0)
    J = np.zeros((n_params, n_obs))

    for i in range(n_params):
        t_p = theta.copy(); t_p[i] += steps[i]
        t_m = theta.copy(); t_m[i] -= steps[i]
        J[i] = (_eval(t_p) - _eval(t_m)) / (2.0 * steps[i])

    range_widths = ranges[:, 1] - ranges[:, 0]
    y0_safe = np.where(np.abs(y0) > 0, y0, 1.0)
    J_norm = J * range_widths[:, np.newaxis] / np.abs(y0_safe)[np.newaxis, :]

    return {
        "jacobian": J,
        "jacobian_normalised": J_norm,
        "theta": theta,
        "y0": y0,
    }
