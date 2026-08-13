"""Posterior calibration metrics.

Implements the three metrics introduced in Gabriel_TLProject:

* **Mahalanobis distance** — squared distance from the true parameter to the
  posterior Gaussian approximation; should be ~ χ²(n_params) distributed
  for a well-calibrated posterior.

* **Energy Score** — proper scoring rule for multivariate distributions;
  lower values indicate a better forecast (proper, parameter-free).

* **KNN log-density** — non-parametric estimate of the posterior density at
  the true parameter value via k-nearest-neighbours.

Reference: Gabriel_TLProject/posterior_evaluation_energysScoreAndMahalonobis.ipynb
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def mahalanobis_distance(
    point: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
) -> float:
    """Squared Mahalanobis distance of ``point`` from Gaussian(mu, cov).

    Parameters
    ----------
    point : array, shape (n,)
    mu : array, shape (n,)
    cov : array, shape (n, n)

    Returns
    -------
    float
        (point - mu)^T @ cov^{-1} @ (point - mu)
    """
    diff = np.asarray(point) - np.asarray(mu)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    return float(diff @ inv_cov @ diff)


def energy_score(
    observation: np.ndarray,
    samples: np.ndarray,
) -> float:
    """Energy Score for a multivariate forecast.

    ES(F, x) = E_F‖X − x‖ − 0.5 E_F‖X − X'‖

    This is a proper scoring rule: the minimum is achieved when the forecast
    distribution F is the true data-generating distribution.

    Parameters
    ----------
    observation : array, shape (n,)
        The observed / target point (e.g. closure-test truth parameters).
    samples : array, shape (N, n)
        Posterior samples representing the forecast distribution.

    Returns
    -------
    float
        Lower is better for the forecast that generated the observation.
    """
    x = np.asarray(observation).reshape(1, -1)
    s = np.asarray(samples)
    term1 = np.mean(np.linalg.norm(s - x, axis=1))
    D = np.linalg.norm(s[:, None, :] - s[None, :, :], axis=2)
    term2 = np.mean(D)
    return float(term1 - 0.5 * term2)


def log_density_knn(
    point: np.ndarray,
    samples: np.ndarray,
    k: Optional[int] = None,
) -> float:
    """KNN-based log-density estimate at ``point`` given posterior ``samples``.

    Uses the k-nearest-neighbour distance estimator of the probability density::

        p̂(x) = k / (N * V_d * r_k^d)

    where ``r_k`` is the distance to the k-th nearest neighbour, ``V_d`` is the
    volume of the unit ball in ``d`` dimensions, and ``N = len(samples)``.

    Parameters
    ----------
    point : array, shape (n,)
    samples : array, shape (N, n)
    k : int, optional
        Number of neighbours; defaults to ``floor(sqrt(N))``.

    Returns
    -------
    float
        Log of the estimated density.
    """
    from sklearn.neighbors import NearestNeighbors

    samples = np.asarray(samples)
    N, d = samples.shape
    k = k or int(np.sqrt(N))
    k = min(max(5, k), N - 1)

    nbrs = NearestNeighbors(n_neighbors=k).fit(samples)
    distances, _ = nbrs.kneighbors(np.atleast_2d(point))
    rk = max(float(distances[0, -1]), 1e-12)

    Vd = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    density = k / (N * Vd * rk ** d)
    return math.log(density)


def calibration_report(
    true_params: np.ndarray,
    flat_chain: np.ndarray,
    param_names: list[str],
) -> dict:
    """Compute all three calibration metrics for a closure test.

    Parameters
    ----------
    true_params : array, shape (n_params,)
        Ground-truth parameter values (the "experiment" design point).
    flat_chain : array, shape (n_samples, n_params)
        Posterior samples.
    param_names : list of str

    Returns
    -------
    dict
        Keys: ``'mahalanobis'``, ``'energy_score'``, ``'log_density'``,
        ``'param_names'``, ``'true_params'``.
    """
    mu = flat_chain.mean(axis=0)
    cov = np.cov(flat_chain.T)
    return {
        "mahalanobis": mahalanobis_distance(true_params, mu, cov),
        "energy_score": energy_score(true_params, flat_chain),
        "log_density": log_density_knn(true_params, flat_chain),
        "param_names": param_names,
        "true_params": np.asarray(true_params).tolist(),
    }
